import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute OUT = X @ W^T + BIAS, where:
      X_ptr: [M, K], row-major (M, K)
      W_ptr: [N, K], row-major (N, K)
      BIAS_ptr: [N]
      OUT_ptr: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K

        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xk)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block as [BLOCK_K, BLOCK_N] for tl.dot(x, w)
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wk + k[:, None] * stride_wm)
        w = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc += tl.dot(x, w)

    # Add bias: broadcast over rows
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, N,
    stride_am, stride_an, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = a * b
    tl.store(OUT_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """
    Compute grouped causal 1D conv:
      Input X: [M, C_in, L], where M = B*S, C_in = H, L = S
      Weight W: [C_in, K] (K=4 for this problem)
      Output Out: [M, C_in, L]
      groups = C_in: per-channel depthwise conv
      No padding (conv1d default), kernel_size = K. For each output time t:
        out[b, c, t] = sum_{k=0..K-1} X[b, c, t - k] * W[c, k], zero when t - k < 0.
    """
    pid_m = tl.program_id(0)  # over M (batch*seq)
    pid_c = tl.program_id(1)  # over output channels (C_in)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_in
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # For each output time position t in tiles, accumulate over kernel window
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # Loop over kernel size K
        for k in range(0, K):
            t_in = t - k  # causal: k in {0,1,2,3} => t_in in [t-3, t], no padding (zero out-of-bounds)
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]

            # Accumulate: acc += x_vals * w_vals[:, None]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c  # bias[c] per output channel
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to Out[b, c, t]
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + offs_t)[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + offs_t)[None, :] < L
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def final_linear_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_N, OUT_N,
    stride_im, stride_in,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute OUT = IN @ W^T + BIAS, where:
      IN_ptr: [M, IN_N], row-major (M, IN_N)
      W_ptr: [OUT_N, IN_N], row-major (OUT_N, IN_N)
      BIAS_ptr: [OUT_N]
      OUT_ptr: [M, OUT_N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_N, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_N

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block as [BLOCK_K, BLOCK_N] for tl.dot(x, w)
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + k[:, None] * stride_wm)
        w = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc += tl.dot(x, w)

    # Add bias: broadcast over rows
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (M_out=3*H, H)
        in_proj_bias: (M_out,)
        conv_weight: (H, 4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be on CUDA for Triton."

        B, S, H = x.shape
        M = B * S

        # 1) in_proj: y_flat = x_flat @ in_proj_weight^T + in_proj_bias
        x_flat = x.reshape(M, H).contiguous()
        M_out = 3 * H
        in_proj_out = torch.empty((M, M_out), dtype=x.dtype, device=x.device)

        in_proj_grid = (triton.cdiv(M, 128), triton.cdiv(M_out, 64))
        in_proj_linear_kernel[in_proj_grid](
            x_flat, in_proj_weight, in_proj_bias, in_proj_out,
            M, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            in_proj_out.stride(0), in_proj_out.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3
        )

        # Reshape to (B, S, 3H) and split into B, C, x_proj
        y = in_proj_out.view(B, S, 3 * H).contiguous()
        B_ = y[:, :, :H].reshape(B * S, H).contiguous()
        C_ = y[:, :, H:2 * H].reshape(B * S, H).contiguous()
        XPRJ_ = y[:, :, 2 * H:].reshape(B * S, H).contiguous()

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B * S, H), dtype=x.dtype, device=x.device)
        mul_elementwise_kernel((triton.cdiv(B * S, 128), triton.cdiv(H, 64)))(B_, XPRJ_, Bx,
                                                                             B * S, H,
                                                                             B_.stride(0), B_.stride(1),
                                                                             XPRJ_.stride(0), XPRJ_.stride(1),
                                                                             BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)

        # 3) Grouped causal conv1d: conv_out = conv1d(Bx, conv_weight, conv_bias, groups=H, kernel_size=4)
        conv_out = torch.empty((B * S, H, S), dtype=x.dtype, device=x.device)
        conv_grid = (triton.cdiv(B * S, 128), triton.cdiv(H, 64), triton.cdiv(S, 128))
        grouped_causal_conv1d_kernel[conv_grid](
            Bx, conv_weight, conv_bias, conv_out,
            B * S, H, S, 4,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=128, num_warps=4, num_stages=3
        )

        # 4) Output gating: y = C * conv_out, then transpose back to (B, H, S)
        #   C_: (B*S, H), conv_out: (B*S, H, S)
        #   For each (b,s,h), y[b,s,h] = C[b,s,h] * conv_out[b,s,h]
        gated = torch.empty((B * S, H, S), dtype=x.dtype, device=x.device)
        # Implement elementwise multiply per tile:
        for h0 in range(0, H, 64):
            h = h0 + tl.arange(0, 64)
            h_mask = h < H
            for s0 in range(0, S, 128):
                s = s0 + tl.arange(0, 128)
                s_mask = s < S
                C_tile = C_.reshape(B * S, H)[..., h0:h0+64].reshape(B * S, 64)  # reshape to (B*S, 64)
                conv_tile = conv_out[:, h0:h0+64, s0:s0+128].reshape(B * S, 64, 128)
                # Triton will broadcast; we call a custom elementwise kernel here with 2D grid:
                mul_elementwise_kernel((triton.cdiv(B * S, 128), triton.cdiv(64, 64)), num_warps=4, num_stages=3)(
                    C_tile, conv_tile, gated[:, h0:h0+64, s0:s0+128],
                    B * S, 64,
                    C_tile.stride(0), C_tile.stride(1),
                    conv_tile.stride(0), conv_tile.stride(1),
                    gated[:, h0:h0+64, s0:s0+128].stride(0), gated[:, h0:h0+64, s0:s0+128].stride(1)
                )
        # Note: The above pattern uses Triton for elementwise multiply in tiles. For simplicity, the original example shows F.conv1d; here we implement it directly.

        # 5) Final projection: y @ out_proj_weight^T + out_proj_bias, output (B, S, H)
        #   We need y of shape (B*S, H). Here we can use final_linear_kernel:
        final_out = torch.empty((B * S, H), dtype=x.dtype, device=x.device)
        final_grid = (triton.cdiv(B * S, 128), triton.cdiv(H, 64))
        final_linear_kernel[final_grid](
            gated.reshape(B * S, H), out_proj_weight, out_proj_bias, final_out,
            B * S, H, H,
            gated.reshape(B * S, H).stride(0), gated.reshape(B * S, H).stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3
        )

        # Reshape to (B, S, H)
        output = final_out.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
