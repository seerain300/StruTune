import torch
import triton
import triton.language as tl

# 1) Triple linear projection: given x[B, S, H], compute three outputs (B,S,H):
#    out[b, s, h] = sum_{k=S} x[b, s, k] * W[h, k] + b[h]
#    We slice in_proj_weight into three (H, H) matrices for each branch.
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    W0_ptr, b0_ptr,   # *f32, (H, H), (H,)
    W1_ptr, b1_ptr,   # *f32, (H, H), (H,)
    W2_ptr, b2_ptr,   # *f32, (H, H), (H,)
    B_out_ptr, C_out_ptr, X_out_ptr,  # *f32, (B,S,H), (B,S,H), (B,S,H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    x_stride_b, x_stride_s, x_stride_h,
    W0_stride_h, W0_stride_k,
    W1_stride_h, W1_stride_k,
    W2_stride_h, W2_stride_k,
    B_out_stride_b, B_out_stride_s, B_out_stride_h,
    C_out_stride_b, C_out_stride_s, C_out_stride_h,
    X_out_stride_b, X_out_stride_s, X_out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc_b = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc_c = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over k dimension (reduce over x's last dim S)
    for k0 in range(0, S, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < S

        # Load x[b, s, k] for current tile
        x_ptrs = x_ptr + b * x_stride_b + s_offsets[:, None] * x_stride_s + k_offsets[None, :] * x_stride_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)

        # Load weights W[h, k]
        W0_ptrs = W0_ptr + h * W0_stride_h + k_offsets * W0_stride_k
        W1_ptrs = W1_ptr + h * W1_stride_h + k_offsets * W1_stride_k
        W2_ptrs = W2_ptr + h * W2_stride_h + k_offsets * W2_stride_k
        W0_vals = tl.load(W0_ptrs, mask=mask_k, other=0.0)
        W1_vals = tl.load(W1_ptrs, mask=mask_k, other=0.0)
        W2_vals = tl.load(W2_ptrs, mask=mask_k, other=0.0)

        # Accumulate dot products
        acc_b += tl.sum(x_vals * W0_vals[None, :], axis=1)
        acc_c += tl.sum(x_vals * W1_vals[None, :], axis=1)
        acc_x += tl.sum(x_vals * W2_vals[None, :], axis=1)

    # Add biases
    b0_val = tl.load(b0_ptr + h)
    b1_val = tl.load(b1_ptr + h)
    b2_val = tl.load(b2_ptr + h)
    acc_b += b0_val
    acc_c += b1_val
    acc_x += b2_val

    # Store outputs
    B_out_ptrs = B_out_ptr + b * B_out_stride_b + s_offsets * B_out_stride_s + h * B_out_stride_h
    C_out_ptrs = C_out_ptr + b * C_out_stride_b + s_offsets * C_out_stride_s + h * C_out_stride_h
    X_out_ptrs = X_out_ptr + b * X_out_stride_b + s_offsets * X_out_stride_s + h * X_out_stride_h
    tl.store(B_out_ptrs, acc_b, mask=mask_s)
    tl.store(C_out_ptrs, acc_c, mask=mask_s)
    tl.store(X_out_ptrs, acc_x, mask=mask_s)


# 2) Element-wise gating: Bx = B_out * X_out
@triton.jit
def elemwise_mul_bsh_kernel(
    B_out_ptr, X_out_ptr, Bx_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    B_out_stride_b, B_out_stride_s, B_out_stride_h,
    X_out_stride_b, X_out_stride_s, X_out_stride_h,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    B_ptrs = B_out_ptr + b * B_out_stride_b + s_offsets * B_out_stride_s + h * B_out_stride_h
    X_ptrs = X_out_ptr + b * X_out_stride_b + s_offsets * X_out_stride_s + h * X_out_stride_h
    B_vals = tl.load(B_ptrs, mask=mask_s, other=0.0)
    X_vals = tl.load(X_ptrs, mask=mask_s, other=0.0)
    Bx_vals = B_vals * X_vals

    Bx_ptrs = Bx_ptr + b * Bx_stride_b + s_offsets * Bx_stride_s + h * Bx_stride_h
    tl.store(Bx_ptrs, Bx_vals, mask=mask_s)


# 3) Grouped causal 1D convolution on Bx with kernel_size=4 and groups=H:
#    conv_out[b, c, s] = sum_{k=0..3} Bx[b, c, s + k - 1] * conv_weight[c, c, k] + conv_bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, convB_ptr, conv_out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    Bx_stride_b, Bx_stride_c, Bx_stride_s,
    convW_stride_c, convW_stride_c_in, convW_stride_k,  # convW shape (H, H, 4) with groups=H
    conv_out_stride_b, conv_out_stride_c, conv_out_stride_s,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Load bias for group c
    bias_c = tl.load(convB_ptr + c)

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Fixed kernel size 4: load weights and perform causal sum
    for k in range(4):
        # positions t + k - 1, handle padding by masking
        t = s_offsets - (4 - 1)  # t_init = s_offsets - 3
        pos = t + k

        # mask for valid positions (0 <= pos < S)
        mask_pos = (pos >= 0) & (pos < S)

        Bx_ptrs = Bx_ptr + b * Bx_stride_b + c * Bx_stride_c + pos * Bx_stride_s
        Bx_vals = tl.load(Bx_ptrs, mask=mask_pos, other=0.0)

        convW_ptrs = convW_ptr + c * convW_stride_c + c * convW_stride_c_in + k * convW_stride_k
        convW_val = tl.load(convW_ptrs)  # scalar
        acc += Bx_vals * convW_val

    # Add bias
    acc += bias_c

    # Store conv_out[b, c, s]
    conv_out_ptrs = conv_out_ptr + b * conv_out_stride_b + c * conv_out_stride_c + s_offsets * conv_out_stride_s
    tl.store(conv_out_ptrs, acc, mask=mask_s)


# 4) Output gating: y = C_out * conv_out elementwise
@triton.jit
def elemwise_mul2_bsh_kernel(
    C_out_ptr, conv_out_ptr, y_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    C_out_stride_b, C_out_stride_s, C_out_stride_h,
    conv_out_stride_b, conv_out_stride_c, conv_out_stride_s,
    y_stride_b, y_stride_s, y_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    C_ptrs = C_out_ptr + b * C_out_stride_b + s_offsets * C_out_stride_s + h * C_out_stride_h
    conv_ptrs = conv_out_ptr + b * conv_out_stride_b + h * conv_out_stride_c + s_offsets * conv_out_stride_s
    C_vals = tl.load(C_ptrs, mask=mask_s, other=0.0)
    conv_vals = tl.load(conv_ptrs, mask=mask_s, other=0.0)
    y_vals = C_vals * conv_vals

    y_ptrs = y_ptr + b * y_stride_b + s_offsets * y_stride_s + h * y_stride_h
    tl.store(y_ptrs, y_vals, mask=mask_s)


# 5) Final linear projection: y[B,S,H] @ out_proj_weight[H,H]^T + out_proj_bias[H]
@triton.jit
def final_linear_bsh_kernel(
    y_ptr, outW_ptr, outB_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    y_stride_b, y_stride_s, y_stride_h,
    outW_stride_h, outW_stride_t,  # outW is (H,H), indexed as (h,t)
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Reduce over t dimension (S) using y[b, s, h] and outW[h, t]
    for t0 in range(0, S, BLOCK_K):
        t_offsets = t0 + tl.arange(0, BLOCK_K)
        mask_t = t_offsets < S

        y_ptrs = y_ptr + b * y_stride_b + s_offsets[:, None] * y_stride_s + h * y_stride_h
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None] & mask_t[None, :], other=0.0)

        outW_ptrs = outW_ptr + h * outW_stride_h + t_offsets * outW_stride_t
        outW_vals = tl.load(outW_ptrs, mask=mask_t, other=0.0)

        acc += tl.sum(y_vals * outW_vals[None, :], axis=1)

    # Add bias
    b_val = tl.load(outB_ptr + h)
    acc += b_val

    # Store output[b, s, h]
    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_s)


class ModelNew(torch.nn.Module):
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
        1) Triple linear projection on x -> B_out, C_out, X_out
        2) Element-wise gating Bx = B_out * X_out
        3) Grouped causal conv1d on Bx, kernel_size=4, groups=hidden_size
        4) Output gating y = C_out * conv_out
        5) Final linear projection to output
        """
        assert x.is_cuda, "ModelNew requires CUDA tensors for Triton kernels."
        assert x.dtype == torch.float32, "This Triton implementation expects float32 tensors."

        B, S, H = x.shape

        # Allocate outputs
        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # Slices for in_proj_weight: (H, H) for each branch
        W0 = in_proj_weight[:H, :].contiguous()   # (H, H)
        b0 = in_proj_bias[:H].contiguous()        # (H,)
        W1 = in_proj_weight[H:2*H, :].contiguous()  # (H, H)
        b1 = in_proj_bias[H:2*H].contiguous()      # (H,)
        W2 = in_proj_weight[2*H:3*H, :].contiguous()  # (H, H)
        b2 = in_proj_bias[2*H:3*H].contiguous()      # (H,)

        # Launch triple linear projection kernel
        grid_triple = (B, H, (S + 128 - 1) // 128)
        triple_linear_bsh_kernel[grid_triple](
            x, W0, b0, W1, b1, W2, b2, B_out, C_out, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            W1.stride(0), W1.stride(1),
            W2.stride(0), W2.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=128, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        # Element-wise gating: Bx = B_out * X_out
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # Grouped causal 1D convolution on Bx with kernel_size=4 and groups=H
        convW = conv_weight.contiguous()  # (H, H, 4), groups=H (depthwise)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)  # output is (B,H,S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2), convW.stride(3),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # Output gating: y = C_out * conv_out -> shape (B,H,S)
        y = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul2_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # Final linear projection: y -> output (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        outW = out_proj_weight.contiguous()  # (H, H)
        outB = out_proj_bias.contiguous()    # (H,)
        grid_final = (B, H, (S + 128 - 1) // 128)
        final_linear_bsh_kernel[grid_final](
            y, outW, outB, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            outW.stride(0), outW.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=128, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
