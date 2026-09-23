import torch
import triton
import triton.language as tl


# 1) Triple linear projection kernel: computes one of the 3 outputs (B, S, H)
#    Using x[B, S, H], weight[H, H], bias[H], produces out[B, S, H]
@triton.jit
def triple_linear_bsh_kernel(
    x_ptr,           # *f32, (B, S, H)
    weight_ptr,      # *f32, (H, H)
    bias_ptr,        # *f32, (H,)
    out_ptr,         # *f32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    x_stride_b, x_stride_s, x_stride_h,
    w_stride_0, w_stride_1,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for output vector of length BLOCK_S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over hidden dimension K=H for the dot product
    for k in range(0, H, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load weight vector for current h across k
        w_vals = tl.load(weight_ptr + pid_h * w_stride_0 + k_offsets * w_stride_1, mask=mask_k, other=0.0)

        # For each s in tile, load x[b, s, k] and accumulate
        # x_ptr indexing: b*stride_b + s*stride_s + k*stride_h
        for kk in range(0, BLOCK_K):
            k_idx = k + kk
            if k_idx < H:
                # Load x[b, s, k_idx] for all s in tile
                x_vec = tl.load(
                    x_ptr + pid_b * x_stride_b + s_offsets * x_stride_s + k_idx * x_stride_h,
                    mask=mask_s,
                    other=0.0
                )
                acc += x_vec * w_vals[kk]

    # Add bias for each s
    b_val = tl.load(bias_ptr + pid_h, mask=True, other=0.0)  # scalar
    acc += b_val

    # Store results
    tl.store(out_ptr + pid_b * out_stride_b + s_offsets * out_stride_s + pid_h * out_stride_h, acc, mask=mask_s)


# 2) Element-wise gating: out = A * B, A,B shape (B,S,H), out shape (B,S,H)
@triton.jit
def elemwise_mul_bsh_kernel(
    A_ptr, B_ptr, Out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    A_stride_b, A_stride_s, A_stride_h,
    B_stride_b, B_stride_s, B_stride_h,
    Out_stride_b, Out_stride_s, Out_stride_h,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    A_vec = tl.load(A_ptr + pid_b * A_stride_b + s_offsets * A_stride_s + pid_h * A_stride_h, mask=mask_s, other=0.0)
    B_vec = tl.load(B_ptr + pid_b * B_stride_b + s_offsets * B_stride_s + pid_h * B_stride_h, mask=mask_s, other=0.0)
    Out_vec = A_vec * B_vec
    tl.store(Out_ptr + pid_b * Out_stride_b + s_offsets * Out_stride_s + pid_h * Out_stride_h, Out_vec, mask=mask_s)


# 3) Grouped causal 1D convolution:
#    Input: Bx (B, S, H), conv_weight (H, H, 4), conv_bias (H)
#    Output: conv_out (B, H, S)
#    conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t+k-1] * conv_weight[c, c, k] + conv_bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,           # *f32, (B, S, H)
    convW_ptr,        # *f32, (H, H, 4) grouped by H
    convB_ptr,        # *f32, (H,)
    conv_out_ptr,     # *f32, (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    Bx_stride_b, Bx_stride_s, Bx_stride_h,
    convW_stride_0, convW_stride_1, convW_stride_2,  # strides for (H, H, 4)
    conv_out_stride_b, conv_out_stride_h, conv_out_stride_s,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # c is the output channel
    pid_s = tl.program_id(2)  # tile over S

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # For causal conv with kernel_size=4:
    # contributions come from Bx[b, c, s-1], Bx[b, c, s], Bx[b, c, s+1], Bx[b, c, s+2]
    # with convW[c, c, 0..3]
    # We compute indices ensuring masks for padding (out-of-range => 0).
    for k in range(4):
        t = s_offsets + k - 1
        # Mask for valid Bx index: t in [0, S-1]
        mask_t = (t >= 0) & (t < S)
        vals = tl.load(Bx_ptr + pid_b * Bx_stride_b + t * Bx_stride_s + pid_c * Bx_stride_h, mask=mask_t & mask_s, other=0.0)
        w_val = tl.load(convW_ptr + pid_c * convW_stride_0 + pid_c * convW_stride_1 + k * convW_stride_2)
        acc += vals * w_val

    # Add bias
    b_val = tl.load(convB_ptr + pid_c, mask=True, other=0.0)
    acc += b_val

    # Store to conv_out[b, c, s]
    tl.store(conv_out_ptr + pid_b * conv_out_stride_b + pid_c * conv_out_stride_h + s_offsets * conv_out_stride_s, acc, mask=mask_s)


# 4) Output gating: y = C_out * conv_out, shapes (B, H, S)
@triton.jit
def elemwise_mul_bhs_kernel(
    A_ptr, B_ptr, Out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    A_stride_b, A_stride_h, A_stride_s,
    B_stride_b, B_stride_h, B_stride_s,
    Out_stride_b, Out_stride_h, Out_stride_s,
    BLOCK_S: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    A_vec = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + s_offsets * A_stride_s, mask=mask_s, other=0.0)
    B_vec = tl.load(B_ptr + pid_b * B_stride_b + pid_h * B_stride_h + s_offsets * B_stride_s, mask=mask_s, other=0.0)
    Out_vec = A_vec * B_vec
    tl.store(Out_ptr + pid_b * Out_stride_b + pid_h * Out_stride_h + s_offsets * Out_stride_s, Out_vec, mask=mask_s)


# 5) Final linear projection: y -> out (B, S, H)
#    y has shape (B, S, H), out_proj_weight (H, H), out_proj_bias (H)
@triton.jit
def final_linear_bsh_kernel(
    y_ptr,           # *f32, (B, S, H)
    W_ptr,           # *f32, (H, H)
    b_ptr,           # *f32, (H,)
    out_ptr,         # *f32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    y_stride_b, y_stride_s, y_stride_h,
    W_stride_0, W_stride_1,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Reduce over H (input feature dimension) for each output h
    for k in range(0, H, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        w_vec = tl.load(W_ptr + pid_h * W_stride_0 + k_offsets * W_stride_1, mask=mask_k, other=0.0)
        for kk in range(0, BLOCK_K):
            k_idx = k + kk
            if k_idx < H:
                y_val = tl.load(y_ptr + pid_b * y_stride_b + pid_s * y_stride_s + k_idx * y_stride_h)
                acc += y_val * w_vec[kk]

    # Add bias
    b_val = tl.load(b_ptr + pid_h, mask=True, other=0.0)
    acc += b_val

    # Store
    tl.store(out_ptr + pid_b * out_stride_b + pid_s * out_stride_s + pid_h * out_stride_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        """
        x: (B, S, H)
        in_proj_weight: (3*H, H)
        in_proj_bias: (3*H,)
        conv_weight: (H, H, 4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        device = x.device
        dtype = x.dtype

        B, S, H = x.shape

        # 1) Triple linear projection
        # Slice in_proj_weight into (H,H) for B, C, x_proj
        W0 = in_proj_weight[:H, :].contiguous()       # (H, H)
        b0 = in_proj_bias[:H].contiguous()           # (H,)
        W1 = in_proj_weight[H:2 * H, :].contiguous() # (H, H)
        b1 = in_proj_bias[H:2 * H].contiguous()      # (H,)
        W2 = in_proj_weight[2 * H:3 * H, :].contiguous()  # (H, H)
        b2 = in_proj_bias[2 * H:3 * H].contiguous()  # (H,)

        # Allocate outputs (B, S, H) for each
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch Triton kernels
        grid_lin = (B, H, (S + 128 - 1) // 128)
        triple_linear_bsh_kernel[grid_lin](
            x, W0, b0, B_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W0.stride(0), W0.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=128, BLOCK_K=64, num_warps=4, num_stages=2
        )
        triple_linear_bsh_kernel[grid_lin](
            x, W1, b1, C_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), W1.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=128, BLOCK_K=64, num_warps=4, num_stages=2
        )
        triple_linear_bsh_kernel[grid_lin](
            x, W2, b2, X_out,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), W2.stride(1),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            BLOCK_S=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) Element-wise gating: Bx = B * X
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

        # 3) Grouped causal 1D convolution
        convW = conv_weight.contiguous()   # (H, H, 4)
        convB = conv_bias.contiguous()     # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)  # output is (B, H, S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> shape (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bhs_kernel[grid_mul2](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, num_warps=4, num_stages=2
        )

        # 5) Final linear projection: y -> (B, S, H)
        out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_fin = (B, S, H)
        final_linear_bsh_kernel[grid_fin](
            y, out_proj_weight.contiguous(), out_proj_bias.contiguous(), out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
