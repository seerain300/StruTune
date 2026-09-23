import torch
import triton
import triton.language as tl

# Kernel 1: Triple linear projection over x[B, S, H] with three output groups (B, C, X), each (B, S, H).
# We take slices from in_proj_weight: W0[:H, :], W1[H:2H, :], W2[2H:3H, :], and biases b0, b1, b2.
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H), contiguous or strided
    W0_ptr, b0_ptr,  # *f32, (H, S) for W0, *f32, (H,) bias
    W1_ptr, b1_ptr,  # *f32, (H, S) for W1
    W2_ptr, b2_ptr,  # *f32, (H, S) for W2
    out_ptr,         # *f32, (B, S, H) final output for this group
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    x_stride_b, x_stride_s, x_stride_h,
    w_stride_h, w_stride_k,  # weight stride for (H, S): W[h, k] with stride (w_stride_h, w_stride_k)
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)  # batch
    h = tl.program_id(1)  # hidden index
    tile_s = tl.program_id(2)  # tile along sequence dimension

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for output vector of length BLOCK_S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over K=S in chunks
    for k0 in range(0, S, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < S

        # Load x[b, s, k] as a vector over s_offsets
        x_ptrs = x_ptr + b * x_stride_b + s_offsets[:, None] * x_stride_s + k_offsets[None, :] * x_stride_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)  # (BLOCK_S, BLOCK_K)

        # Load corresponding weights W[h, k] as a vector over k_offsets
        w_ptrs = W0_ptr + h * w_stride_h + k_offsets * w_stride_k
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)  # (BLOCK_K,)

        # Accumulate dot product over k
        acc += tl.sum(x_vals * w_vals[None, :], axis=1)  # (BLOCK_S,)

    # Add bias
    b_val = tl.load(b0_ptr + h)  # scalar
    acc += b_val

    # Store to out[b, s, h]
    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_s)


# Kernel 2: Elementwise multiply for (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    a_stride_b, a_stride_s, a_stride_h,
    b_stride_b, b_stride_s, b_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    a_ptrs = a_ptr + b * a_stride_b + s_offsets * a_stride_s + h * a_stride_h
    b_ptrs = b_ptr + b * b_stride_b + s_offsets * b_stride_s + h * b_stride_h

    a_vals = tl.load(a_ptrs, mask=mask_s, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask_s, other=0.0)

    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, a_vals * b_vals, mask=mask_s)


# Kernel 3: Grouped causal 1D convolution with kernel_size=4 and groups=H.
# Input Bx has shape (B, S, H). conv_weight has shape (H, H, 4), groups=H, bias (H,).
# Output conv_out has shape (B, H, S).
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *f32, (B, S, H) input after gating
    convW_ptr,     # *f32, (H, H, 4) filters, groups=H
    convB_ptr,     # *f32, (H,) bias
    out_ptr,       # *f32, (B, H, S) output
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride_g, convW_stride_c, convW_stride_k,  # (g, c, k) strides
    out_stride_b, out_stride_c, out_stride_s,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)  # batch
    c = tl.program_id(1)  # output channel (group index)
    tile_s = tl.program_id(2)  # tiles over output positions t

    t_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_t = t_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # kernel_size=4, causal: output t depends on Bx[b, c, t + k - 1] for k in [0..3]
    for k in range(4):
        t_pad = t_offsets + k - 1  # left causal padding
        mask_pos = (t_pad >= 0) & (t_pad < S) & mask_t

        # Load input positions Bx[b, c, t_pad]
        Bx_ptrs = Bx_ptr + b * Bx_stride_b + t_pad * Bx_stride_s + c * Bx_stride_h
        Bx_vals = tl.load(Bx_ptrs, mask=mask_pos, other=0.0)  # (BLOCK_S,)

        # Load filter weight convW[c, c, k] (since groups=H, group index is c)
        # convW has shape (H, H, 4) with strides (convW_stride_g, convW_stride_c, convW_stride_k)
        convW_val = tl.load(convW_ptr + c * convW_stride_g + c * convW_stride_c + k * convW_stride_k)

        acc += Bx_vals * convW_val

    # Add bias
    b_val = tl.load(convB_ptr + c)
    acc += b_val

    # Store to conv_out[b, c, t] = acc
    out_ptrs = out_ptr + b * out_stride_b + c * out_stride_c + t_offsets * out_stride_s
    tl.store(out_ptrs, acc, mask=mask_t)


# Kernel 4: Linear over (B, S, H) using weight (H, H) and bias (H).
@triton.jit
def linear_bsh_kernel(
    in_ptr,        # *f32, (B, S, H) input
    W_ptr, b_ptr,  # *f32, (H, H) weight, *f32, (H,) bias
    out_ptr,       # *f32, (B, S, H) output
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    in_stride_b, in_stride_s, in_stride_h,
    W_stride_h, W_stride_k,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)  # batch
    h = tl.program_id(1)  # output hidden index
    tile_s = tl.program_id(2)  # tile over S

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over K=S in chunks
    for k0 in range(0, S, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < S

        # Load in[b, s, k] as a vector over s_offsets
        in_ptrs = in_ptr + b * in_stride_b + s_offsets[:, None] * in_stride_s + k_offsets[None, :] * in_stride_h
        in_vals = tl.load(in_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)  # (BLOCK_S, BLOCK_K)

        # Load W[h, k] as vector over k_offsets
        W_ptrs = W_ptr + h * W_stride_h + k_offsets * W_stride_k
        W_vals = tl.load(W_ptrs, mask=mask_k, other=0.0)  # (BLOCK_K,)

        # Accumulate dot over k
        acc += tl.sum(in_vals * W_vals[None, :], axis=1)  # (BLOCK_S,)

    # Add bias
    b_val = tl.load(b_ptr + h)
    acc += b_val

    # Store to out[b, s, h]
    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_s)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Triton-optimized fused pipeline:
        1) Triple linear projection on x -> B, C, x_proj
        2) Element-wise gating Bx = B * x_proj
        3) Grouped causal conv1d on Bx, kernel_size=4, groups=hidden_size
        4) Output gating y = C * conv_out
        5) Final linear projection to output
        """
        assert x.is_cuda, "ModelNew requires CUDA tensors for Triton kernels."
        device = x.device
        B, S, H = x.shape

        # 1) Triple linear projection: B, C, X_out
        # Make x strided (B, S, H) without transposing
        x_contig = x.contiguous()

        # W0 slice: (H, H), bias b0: (H,)
        W0 = in_proj_weight[:H, :].contiguous()
        b0 = in_proj_bias[:H].contiguous()

        # W1 slice: (H, H), bias b1: (H,)
        W1 = in_proj_weight[H:2 * H, :].contiguous()
        b1 = in_proj_bias[H:2 * H].contiguous()

        # W2 slice: (H, H), bias b2: (H,)
        W2 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b2 = in_proj_bias[2 * H:3 * H].contiguous()

        # Outputs
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch triple_linear_bsh_kernel for each group
        grid1 = (B, H, (S + 128 - 1) // 128)
        triple_linear_bsh_kernel[grid1](
            x_contig, W0, b0, W1, b1, X_out,
            B, S, H,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            W0.stride(0), W0.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        grid2 = (B, H, (S + 128 - 1) // 128)
        triple_linear_bsh_kernel[grid2](
            x_contig, W0, b0, W1, b1, C_out,
            B, S, H,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        grid3 = (B, H, (S + 128 - 1) // 128)
        triple_linear_bsh_kernel[grid3](
            x_contig, W0, b0, W2, b2, B_out,
            B, S, H,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 3) Grouped causal 1D conv: conv_out[b, c, s] using Bx
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)  # output (B,H,S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> shape (B,H,S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 5) Final linear projection to (B, S, H)
        # y is (B, H, S); we compute out[b, s, h] = sum_k y[b, s, k] * out_proj_weight[h, k] + bias[h]
        y_trans = y.transpose(-1, -2).contiguous()  # (B, S, H)
        out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid5 = (B, H, (S + 128 - 1) // 128)
        linear_bsh_kernel[grid5](
            y_trans, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y_trans.stride(0), y_trans.stride(1), y_trans.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
