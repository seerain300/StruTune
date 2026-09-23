import torch
import triton
import triton.language as tl


# 1) Linear projection: given x[B, S, H] and weight[M, H] (M = 3*H), compute out[B, S, M]
#    out[b, s, m] = sum_k x[b, s, k] * weight[m, k] + bias[m]
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
    b_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    m_idx = tl.program_id(2)

    # Accumulator for output value
    acc = tl.zeros((), dtype=tl.float32)

    # Reduce over H in chunks
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load x[b, s, k_offsets]
        x_ptrs = x_ptr + b_idx * x_stride0 + s_idx * x_stride1 + k_offsets * x_stride2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)  # shape (BLOCK_K,)

        # Load weight[m, k_offsets]
        w_ptrs = weight_ptr + m_idx * weight_stride0 + k_offsets * weight_stride1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)  # shape (BLOCK_K,)

        # Fused multiply and sum across BLOCK_K
        acc += tl.sum(x_vals * w_vals, axis=0)

    # Add bias[m]
    b_val = tl.load(bias_ptr + m_idx)
    acc += b_val

    # Store out[b, s, m]
    out_ptrs = out_ptr + b_idx * out_stride0 + s_idx * out_stride1 + m_idx * out_stride2
    tl.store(out_ptrs, acc)


# 2) Element-wise gating: Bx = B * X, where B and X are (B, H, S). Produces (B, H, S).
@triton.jit
def elemwise_mul_bsh_kernel(
    B_ptr, X_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    B_stride0, B_stride1, B_stride2,
    X_stride0, X_stride1, X_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    tile_s = tl.program_id(2)
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    B_ptrs = B_ptr + b_idx * B_stride0 + h_idx * B_stride1 + s_offsets * B_stride2
    X_ptrs = X_ptr + b_idx * X_stride0 + h_idx * X_stride1 + s_offsets * X_stride2
    out_ptrs = out_ptr + b_idx * out_stride0 + h_idx * out_stride1 + s_offsets * out_stride2

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
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    tile_s = tl.program_id(2)

    t_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_t = t_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # k loop over 4 taps; causal left padding only
    for k in range(4):
        s_idx = t_offsets + k - 1
        valid = s_idx >= 0
        Bx_ptrs = Bx_ptr + b_idx * Bx_stride0 + s_idx * Bx_stride1 + c_idx * Bx_stride2
        Bx_vals = tl.load(Bx_ptrs, mask=mask_t & valid, other=0.0)
        convW_val = tl.load(convW_ptr + c_idx * convW_stride0 + c_idx * convW_stride1 + k * convW_stride2)
        acc += Bx_vals * convW_val

    # Add bias
    convB_val = tl.load(convB_ptr + c_idx)
    acc += convB_val

    out_ptrs = out_ptr + b_idx * out_stride0 + c_idx * out_stride1 + t_offsets * out_stride2
    tl.store(out_ptrs, acc, mask=mask_t)


# 4) Final linear projection: y[B, H, S], out_proj_weight[H, H], out_proj_bias[H] -> out[B, S, H]
#    out[b, s, h] = sum_c y[b, h, s] * out_proj_weight[h, c] + out_proj_bias[h]
@triton.jit
def final_linear_bhs_to_bsh_kernel(
    y_ptr,           # *f32, (B, H, S)
    out_proj_ptr,    # *f32, (H, H)
    out_proj_bias_ptr,  # *f32, (H,)
    out_ptr,         # *f32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    y_stride0, y_stride1, y_stride2,
    out_proj_stride0, out_proj_stride1,
    out_stride0, out_stride1, out_stride2,
    BLOCK_S: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    tile_s = tl.program_id(2)
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator per s_offsets
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Reduce over channels c = 0..H-1
    for c in range(0, H):
        # y[b, h_idx, s_offsets] * out_proj[h_idx, c]
        y_ptrs = y_ptr + b_idx * y_stride0 + h_idx * y_stride1 + s_offsets * y_stride2
        y_vals = tl.load(y_ptrs, mask=mask_s, other=0.0)

        out_proj_val = tl.load(out_proj_ptr + h_idx * out_proj_stride0 + c * out_proj_stride1)
        acc += y_vals * out_proj_val

    # Add bias
    bias_val = tl.load(out_proj_bias_ptr + h_idx)
    acc += bias_val

    # Store to out[b, s_offsets, h_idx]
    out_ptrs = out_ptr + b_idx * out_stride0 + s_offsets * out_stride1 + h_idx * out_stride2
    tl.store(out_ptrs, acc, mask=mask_s)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure dtype is float32 for Triton
        device = x.device
        dtype = torch.float32
        x = x.to(dtype).contiguous()
        in_proj_weight = in_proj_weight.to(dtype).contiguous()
        in_proj_bias = in_proj_bias.to(dtype).contiguous()
        conv_weight = conv_weight.to(dtype).contiguous()
        conv_bias = conv_bias.to(dtype).contiguous()
        out_proj_weight = out_proj_weight.to(dtype).contiguous()
        out_proj_bias = out_proj_bias.to(dtype).contiguous()

        B, S, H = x.shape
        M = 3 * H  # in_proj produces 3*H channels

        # 1) Triple linear projection using a single Triton kernel for clarity (we'll create BCx as (B,S,3H))
        #    However, since the original code chunks after transpose, we can directly produce BCx via the linear kernel.
        BCx = torch.empty((B, S, M), device=device, dtype=dtype)
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

        # 2) Transpose for chunking into (B, H, S) components: original code does BCx.transpose(-1, -2)
        #    But to match original flow precisely, we can instead slice along the last dimension (3*H) and avoid transpose.
        #    Original code does: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> shape (B, S, 3H)
        #    Then transpose BCx to (B, 3H, S) and chunk into B, C, x_proj, each (B, H, S).
        #    We'll emulate by slicing (B,S,3H) directly without explicit transpose:
        #    Build B, C, X via linear with sliced weights. To keep it single kernel, we'd need 3 calls; but here BCx is (B,S,3H).
        #    Instead, we'll implement chunks using slicing from BCx without relying on transpose:
        #    This is not available in Triton here; so we'll do it in PyTorch after kernel: B = BCx[:, :, :H], etc.
        #    For correctness, we can reconstruct B, C, X via linear with sliced weights in Triton. But to keep a single kernel,
        #    we'll assume in_proj_weight is provided as 3*H already and slice on host.

        # Create slices for B, C, X using host ops (safe, not computation-heavy)
        B_bsh = BCx[:, :, :H].contiguous()  # (B, S, H)
        C_bsh = BCx[:, :, H:2*H].contiguous()  # (B, S, H)
        X_bsh = BCx[:, :, 2*H:].contiguous()  # (B, S, H)

        # 3) Element-wise gating: Bx = B * X
        Bx = torch.empty((B, H, S), device=device, dtype=dtype)
        grid_mul = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul](
            B_bsh, X_bsh, Bx,
            B, H, S,
            B_bsh.stride(0), B_bsh.stride(1), B_bsh.stride(2),
            X_bsh.stride(0), X_bsh.stride(1), X_bsh.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution: conv_out[b, c, s] with kernel_size=4, groups=H
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)  # (B, H, S)
        grid_conv = (B, H, (S + 128 - 1) // 128)
        grouped_causal_conv1d_bsh_to_bhs_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            convW.stride(0), convW.stride(1), convW.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C * conv_out -> shape (B,H,S)
        y_bhs = torch.empty((B, H, S), device=device, dtype=dtype)
        # C_bsh shape is (B, S, H); transpose to (B, H, S) for multiplication
        C_bhs = C_bsh.transpose(1, 2).contiguous()  # (B, H, S)
        grid_mul2 = (B, H, (S + 128 - 1) // 128)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_bhs, conv_out, y_bhs,
            B, H, S,
            C_bhs.stride(0), C_bhs.stride(1), C_bhs.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y_bhs.stride(0), y_bhs.stride(1), y_bhs.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        # 6) Final linear projection: y_bhs (B,H,S) -> out (B,S,H)
        out_bsh = torch.empty((B, S, H), device=device, dtype=dtype)
        grid_final = (B, H, (S + 128 - 1) // 128)
        final_linear_bhs_to_bsh_kernel[grid_final](
            y_bhs, out_proj_weight, out_proj_bias, out_bsh,
            B, S, H,
            y_bhs.stride(0), y_bhs.stride(1), y_bhs.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_bsh.stride(0), out_bsh.stride(1), out_bsh.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2
        )

        return out_bsh


def run(*args):
    return ModelNew()(*args)
