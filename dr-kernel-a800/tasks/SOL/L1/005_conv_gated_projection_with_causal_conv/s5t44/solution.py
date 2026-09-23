import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, K, N,  # M=B*S, K=H, N=3*H
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    OUT = X @ W^T + BIAS
    X: [M, K], row-major (M, K) with strides (stride_xm, stride_xk)
    W: [N, K], row-major (N, K) with strides (stride_wm, stride_wk)
    BIAS: [N]
    OUT: [M, N] with strides (stride_om, stride_ok)
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

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, K,  # M=B*S, K=H
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=m_mask[:, None], other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk, mask=m_mask[:, None], other=0.0)
    out = a * b

    tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, out, mask=m_mask[:, None])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,  # M=B*S, C_in=H, L=S, K=4
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """
    Grouped causal 1D conv with zero padding:
      Input X: [M, C_in, L] (we pass as strided tensor), M=B*S
      Weight W: [C_in, K]
      Output Out: [M, C_in, L]
      groups = C_in: per-channel depthwise conv
      No padding (conv1d default). For output t, we use input at t - k, masked when t - k < 0.
    """
    pid_m = tl.program_id(0)  # over M
    pid_c = tl.program_id(1)  # over output channels (C_in)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_in
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Tile over time positions
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # current time tile
        ti_mask = t < L

        # Accumulate over kernel window k in {0,1,2,3}
        for k in range(0, K):
            t_in = t - k  # no padding: zero out-of-bounds via mask
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
    bias_ptrs = BIAS_ptr + offs_c  # bias[h] per output channel
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to Out[b, c, t]
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + t[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t[None, :] < L)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def final_proj_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute OUT = IN @ W^T + BIAS, where:
      IN: [M, IN_H], row-major (M, IN_H)
      W: [OUT_H, IN_H], row-major (OUT_H, IN_H)
      BIAS: [OUT_H]
      OUT: [M, OUT_H]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H

        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = W_ptr + (offs_n[None, :] * stride_wh + k[:, None] * stride_wm)
        w = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc += tl.dot(x, w)

    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

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
        # Flatten x to (M, H)
        x_flat = x.reshape(M, H).contiguous()
        in_proj_out = torch.empty((M, 3 * H), dtype=x.dtype, device=x.device)
        in_proj_weight_ = in_proj_weight.contiguous()
        in_proj_bias_ = in_proj_bias.contiguous()

        grid_in = (triton.cdiv(M, 128), triton.cdiv(3 * H, 64))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight_, in_proj_bias_, in_proj_out,
            M, H, 3 * H,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight_.stride(0), in_proj_weight_.stride(1),
            in_proj_out.stride(0), in_proj_out.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
        )

        # Split y_flat into B, C, x_proj along last dimension without torch ops
        # y_flat: (M, 3H) -> B: (M, H), C: (M, H), x_proj: (M, H)
        M_ = M
        H_ = H
        B_flat = in_proj_out[:, :H]
        C_flat = in_proj_out[:, H:2 * H]
        x_proj_flat = in_proj_out[:, 2 * H:]

        # 2) Element-wise gating: Bx = B * x_proj
        Bx_flat = torch.empty((M, H), dtype=x.dtype, device=x.device)
        grid_mul = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        mul_elementwise_kernel[grid_mul](
            B_flat, x_proj_flat, Bx_flat,
            M, H,
            B_flat.stride(0), B_flat.stride(1),
            x_proj_flat.stride(0), x_proj_flat.stride(1),
            Bx_flat.stride(0), Bx_flat.stride(1),
            BLOCK_M=128, BLOCK_K=64,
        )

        # Reshape back to (B, S, H)
        Bx = Bx_flat.reshape(B, S, H).contiguous()

        # 3) Grouped causal 1D convolution: conv with kernel_size=4, groups=H
        # Input X: (M=B*S, C_in=H, L=S), Weight W: (C_in=H, K=4), Output Out: (M, C_in, L)
        X_conv = Bx  # (B,S,H) -> (M,H,S)
        Out_conv = torch.empty((M, H, S), dtype=x.dtype, device=x.device)

        grid_conv = (triton.cdiv(M, 128), triton.cdiv(H, 32), triton.cdiv(S, 64))
        grouped_causal_conv1d_kernel[grid_conv](
            X_conv, conv_weight, conv_bias, Out_conv,
            M, H, S, 4,
            X_conv.stride(0), X_conv.stride(1), X_conv.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            Out_conv.stride(0), Out_conv.stride(1), Out_conv.stride(2),
            BLOCK_M=128, BLOCK_C=32, BLOCK_T=64,
        )

        # Reshape conv_out to (B, H, S)
        conv_out = Out_conv.reshape(B, H, S).contiguous()

        # 4) Output gating: y = C * conv_out
        y_gated = C_flat.view(B, S, H) * conv_out  # (B,S,H)

        # 5) Final projection: y @ out_proj_weight^T + out_proj_bias
        # Flatten y to (M, H)
        y_flat_final = y_gated.reshape(M, H).contiguous()
        out_final = torch.empty((M, H), dtype=x.dtype, device=x.device)
        grid_final = (triton.cdiv(M, 128), triton.cdiv(H, 64), triton.cdiv(H, 32))
        final_proj_kernel[grid_final](
            y_flat_final, out_proj_weight, out_proj_bias, out_final,
            M, H, H,
            y_flat_final.stride(0), y_flat_final.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_final.stride(0), out_final.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
        )

        return out_final.reshape(B, S, H)


def run(*args):
    return ModelNew()(*args)
