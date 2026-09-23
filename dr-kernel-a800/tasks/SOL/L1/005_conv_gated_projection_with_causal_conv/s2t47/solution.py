import torch
import triton
import triton.language as tl

# 1) Linear projection: compute out[B, S, M] = x @ weight.T + bias
#    x: (B, S, H), weight: (M, H), bias: (M,), out: (B, S, M)
@triton.jit
def linear_bsh_kernel(
    x_ptr,         # *f32, (B, S, H)
    weight_ptr,    # *f32, (M, H)
    bias_ptr,      # *f32, (M,)
    out_ptr,       # *f32, (B, S, M)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
    x_stride0, x_stride1, x_stride2,
    weight_stride0, weight_stride1,
    out_stride0, out_stride1, out_stride2,
    BLOCK_K: tl.constexpr
):
    b_idx = tl.program_id(0)   # 0..B-1
    s_idx = tl.program_id(1)   # 0..S-1
    m_idx = tl.program_id(2)   # 0..M-1

    acc = tl.zeros((), dtype=tl.float32)
    # Reduce over H in chunks of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        # Load x[b, s, k_offsets]
        x_ptrs = x_ptr + b_idx * x_stride0 + s_idx * x_stride1 + k_offsets * x_stride2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)  # shape (BLOCK_K,)
        # Load weight[m, k_offsets] -> (BLOCK_K,)
        w_ptrs = weight_ptr + m_idx * weight_stride0 + k_offsets * weight_stride1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        # Accumulate dot product
        acc += tl.sum(x_vals * w_vals, axis=0)
    # Add bias
    b_val = tl.load(bias_ptr + m_idx)
    acc += b_val
    # Store to out[b, s, m]
    out_ptrs = out_ptr + b_idx * out_stride0 + s_idx * out_stride1 + m_idx * out_stride2
    tl.store(out_ptrs, acc)


# 2) Element-wise gating: out[b, s, h] = B[b, s, h] * X[b, s, h]
@triton.jit
def elemwise_mul_bsh_kernel(
    B_ptr, X_ptr, out_ptr,  # *f32, (B, S, H)
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    B_stride0, B_stride1, B_stride2,
    X_stride0, X_stride1, X_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr
):
    b_idx = tl.program_id(0)   # 0..B-1
    h_idx = tl.program_id(1)   # 0..H-1
    tile_s = tl.program_id(2)  # tiles over S
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    B_ptrs = B_ptr + b_idx * B_stride0 + s_offsets * B_stride1 + h_idx * B_stride2
    X_ptrs = X_ptr + b_idx * X_stride0 + s_offsets * X_stride1 + h_idx * X_stride2
    out_ptrs = out_ptr + b_idx * out_stride0 + s_offsets * out_stride1 + h_idx * out_stride2

    B_vals = tl.load(B_ptrs, mask=mask_s, other=0.0)
    X_vals = tl.load(X_ptrs, mask=mask_s, other=0.0)
    tl.store(out_ptrs, B_vals * X_vals, mask=mask_s)


# 3) Grouped causal 1D convolution:
#    Input Bx: (B, S, H), convW: (H, H, 4), convB: (H,)
#    Output conv_out: (B, H, S)
#    For each (b, c, t): conv_out[b, c, t] = sum_{k=0..3} Bx[b, c, t+k-1] * convW[c, c, k] + convB[c]
@triton.jit
def grouped_causal_conv1d_bsh_to_bhs_kernel(
    Bx_ptr,         # *f32, (B, S, H)
    convW_ptr,      # *f32, (H, H, 4)
    convB_ptr,      # *f32, (H,)
    out_ptr,        # *f32, (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    Bx_stride0, Bx_stride1, Bx_stride2,
    convW_stride0, convW_stride1, convW_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr
):
    b_idx = tl.program_id(0)   # 0..B-1
    c_idx = tl.program_id(1)   # 0..H-1
    tile_s = tl.program_id(2)  # tiles over S

    t_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_t = t_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Left padding for kernel_size=4 with causal (only k-1 shift). Pad with zeros if t+k-1 < 0.
    for k in range(4):
        t_shifted = t_offsets + (k - 1)
        valid = t_shifted >= 0
        Bx_ptrs = Bx_ptr + b_idx * Bx_stride0 + t_shifted * Bx_stride1 + c_idx * Bx_stride2
        Bx_vals = tl.load(Bx_ptrs, mask=mask_t & valid, other=0.0)
        convW_val = tl.load(convW_ptr + c_idx * convW_stride0 + c_idx * convW_stride1 + k * convW_stride2)
        acc += Bx_vals * convW_val

    # Add bias
    convB_val = tl.load(convB_ptr + c_idx)
    acc += convB_val

    # Store conv_out[b, c, t] in (B, H, S)
    out_ptrs = out_ptr + b_idx * out_stride0 + c_idx * out_stride1 + t_offsets * out_stride2
    tl.store(out_ptrs, acc, mask=mask_t)


# 4) Final linear projection: y[B, H, S] = y @ out_proj_weight.T + bias
#    y: (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H,)
#    Output: (B, S, H)
@triton.jit
def linear_bhs_to_bsh_kernel(
    y_ptr,          # *f32, (B, H, S)
    out_proj_weight_ptr,  # *f32, (H, H)
    out_proj_bias_ptr,    # *f32, (H,)
    out_ptr,        # *f32, (B, S, H)
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    y_stride0, y_stride1, y_stride2,
    out_proj_weight_stride0, out_proj_weight_stride1,
    out_stride0, out_stride1, out_stride2,
    BLOCK_K: tl.constexpr
):
    b_idx = tl.program_id(0)   # 0..B-1
    s_idx = tl.program_id(1)   # 0..S-1
    h_out = tl.program_id(2)   # 0..H-1

    acc = tl.zeros((), dtype=tl.float32)
    # Reduce over H_in (channels) of y -> output H_out
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        # y[b, k, s]
        y_ptrs = y_ptr + b_idx * y_stride0 + k_offsets * y_stride1 + s_idx * y_stride2
        y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # (BLOCK_K,)
        # out_proj_weight[h_out, k_offsets]
        w_ptrs = out_proj_weight_ptr + h_out * out_proj_weight_stride0 + k_offsets * out_proj_weight_stride1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(y_vals * w_vals, axis=0)

    # Add bias
    bias_val = tl.load(out_proj_bias_ptr + h_out)
    acc += bias_val

    # Store out[b, s, h_out]
    out_ptrs = out_ptr + b_idx * out_stride0 + s_idx * out_stride1 + h_out * out_stride2
    tl.store(out_ptrs, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous
        device = x.device
        x = x.to(torch.float32).contiguous()
        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        B, S, H = x.shape
        M = 3 * H  # in_proj_weight shape (M, H)

        # 1) Compute BCx: (B, S, M) using linear_bsh_kernel
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)
        grid_linear = (B, S, M)
        linear_bsh_kernel[grid_linear](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 2) Slice into B, C, x_proj (B, S, H) each
        B_bsh = BCx[:, :, :H].contiguous()
        C_bsh = BCx[:, :, H:2*H].contiguous()
        X_bsh = BCx[:, :, 2*H:].contiguous()

        # 3) Element-wise gating: Bx = B * X -> (B, S, H)
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul1 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul1](
            B_bsh, X_bsh, Bx,
            B, H, S,
            B_bsh.stride(0), B_bsh.stride(1), B_bsh.stride(2),
            X_bsh.stride(0), X_bsh.stride(1), X_bsh.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_bsh_to_bhs_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C * conv_out -> (B, H, S)
        y_bhs = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_bsh, conv_out, y_bhs,
            B, H, S,
            C_bsh.stride(0), C_bsh.stride(1), C_bsh.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y_bhs.stride(0), y_bhs.stride(1), y_bhs.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 6) Final linear projection: output (B, S, H)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_final = (B, S, H)
        linear_bhs_to_bsh_kernel[grid_final](
            y_bhs, out_proj_weight, out_proj_bias, output,
            B, H, S,
            y_bhs.stride(0), y_bhs.stride(1), y_bhs.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Return result in float32
        return output


def run(*args):
    return ModelNew()(*args)
