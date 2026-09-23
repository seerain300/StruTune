import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Compute OUT = X @ W^T + bias, where:
      X: [M, H], M = B*S
      W: [M_OUT, H], M_OUT = 3H
      OUT: [M, M_OUT]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Accumulate over H
    for k in range(0, H):
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k * stride_xn)
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k * stride_wn)
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)  # [BLOCK_M, 1]
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)  # [BLOCK_N, 1]
        acc += x * w.T

    # Add bias
    bias_vals = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def elementwise_mul_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Compute OUT = A * B elementwise for A, B, OUT shaped [M, N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)

    a = tl.load(a_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    out = a * b

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """
    Grouped causal 1D conv:
      Input X: [M, C_in, L], M=B*S, C_in=H, L=S
      Weight W: [C_in, K], conv per output channel c
      Output OUT: [M, C_in, L]
      Padding is zero (no right-pad), default for conv1d.
    """
    pid_m = tl.program_id(0)  # tile over M
    pid_c = tl.program_id(1)  # tile over channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Iterate over output time positions in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        ti_mask = t < L

        # For each kernel position k in [0, K)
        for k in range(0, K):
            t_in = t - k  # causal: with zero padding for out-of-range
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in]
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k]
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]

            # Accumulate
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]
    acc = acc + bias_vals[None, :]

    # Store Out[b, c, t] for this t tile
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + tl.arange(0, BLOCK_T))[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & ((t0 + tl.arange(0, BLOCK_T))[None, :] < L)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def final_linear_gemm_bias_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute OUT = IN @ W^T + bias
      IN:  [M, IN_H]
      W:   [OUT_H, IN_H]  (note: here OUT_H == IN_H == H)
      OUT: [M, OUT_H]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < IN_H

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K] (W is [OUT_H, IN_H])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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
        Triton-only implementation of the provided run() logic.
        """
        B, S, H = x.shape
        M = B * S

        # 1) Triple linear projection: y_flat = X_flat @ W^T + bias
        X_flat = x.reshape(M, H).contiguous()
        M_OUT = 3 * H
        y_flat = torch.empty((M, M_OUT), device=x.device, dtype=x.dtype)

        grid0 = (triton.cdiv(M, 128), triton.cdiv(M_OUT, 64))
        in_proj_linear_kernel[grid0](
            X_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_OUT,
            X_flat.stride(0), X_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64,
        )

        # Reshape y_flat to (B, S, 3H) and chunk into B, C, x_proj
        y = y_flat.view(B, S, M_OUT)
        B_tensor, C_tensor, x_proj_tensor = torch.chunk(y, 3, dim=1)

        # 2) Element-wise gating: Bx = B * x_proj
        Bx_flat = torch.empty((M, H), device=x.device, dtype=x.dtype)
        elementwise_mul_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 64))](  # grid for (M, H)
            B_tensor.reshape(M, H).contiguous(), x_proj_tensor.reshape(M, H).contiguous(),
            Bx_flat,
            M, H,
            B_tensor.reshape(M, H).stride(0), B_tensor.reshape(M, H).stride(1),
            x_proj_tensor.reshape(M, H).stride(0), x_proj_tensor.reshape(M, H).stride(1),
            Bx_flat.stride(0), Bx_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64,
        )

        # 3) Grouped causal 1D convolution: Out_conv[b, h, t] = sum_{k=0..3} Bx[b,h,t-k] * conv_weight[h,k] + conv_bias[h]
        # Prepare X_conv: (M=B*S, H, S)
        X_conv = Bx_flat.view(M, H, S).contiguous()
        W_conv = conv_weight  # (H, K=4)
        Bias_conv = conv_bias  # (H,)
        Out_conv = torch.empty((M, H, S), device=x.device, dtype=x.dtype)

        grid_conv = (triton.cdiv(M, 128), triton.cdiv(H, 64), triton.cdiv(S, 128))
        grouped_causal_conv1d_kernel[grid_conv](
            X_conv, W_conv, Bias_conv, Out_conv,
            M, H, S, 4,
            X_conv.stride(0), X_conv.stride(1), X_conv.stride(2),
            W_conv.stride(0), W_conv.stride(1),
            Out_conv.stride(0), Out_conv.stride(1), Out_conv.stride(2),
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=128,
        )

        # 4) Output gating: y = C * Out_conv
        C_flat = C_tensor.reshape(M, H).contiguous


def run(*args):
    return ModelNew()(*args)
