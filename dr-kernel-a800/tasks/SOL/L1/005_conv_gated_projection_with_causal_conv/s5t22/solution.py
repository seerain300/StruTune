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
    # X_ptr: [M, H], W_ptr: [M_OUT, H], OUT_ptr: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Accumulate over H
    for k in range(0, H):
        x_vals = tl.load(X_ptr + offs_m[:, None] * stride_xm + k * stride_xn, mask=m_mask[:, None], other=0.0)
        w_vals = tl.load(W_ptr + offs_n[None, :] * stride_wm + k * stride_wn, mask=n_mask[None, :], other=0.0)
        acc += x_vals * w_vals[None, :]

    # Add bias
    bias_vals = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store
    tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def elementwise_mul_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # A_ptr, B_ptr: [M, N], OUT_ptr: [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    c = a * b
    tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, c, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_in, L] where M=B*S, C_in=H, L=S
    # W_ptr: [C_in, K]
    # Out_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Loop over output positions t in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # For each kernel position k, compute input time t_in = t - k (no padding, zero out of bounds)
        for k in range(0, K):
            t_in = t - k  # causal: k in {0,1,2,3} => t_in in [t-3, t]
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
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + offs_t)[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + offs_t)[None, :] < L
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
    # IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
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
        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, w)  # [BLOCK_M, BLOCK_N]

    # Add bias
    bias_vals = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H)
        in_proj_bias: (3H,)
        conv_weight: (H, 4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        Returns: (B, S, H)
        """
        assert x.ndim == 3, "x must be (B, S, H)"
        B, S, H = x.shape
        M = B * S
        M_OUT = in_proj_weight.shape[0]  # 3H
        K = conv_weight.shape[1]  # 4

        device = x.device
        dtype = x.dtype

        # 1) in_proj: y_flat = F.linear(x, in_proj_weight, in_proj_bias) -> (M, 3H)
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, M_OUT), device=device, dtype=dtype)

        grid_in = (triton.cdiv(M, 128), triton.cdiv(M_OUT, 64))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_OUT,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64,
        )

        # Reshape to (B, S, 3H) and chunk into B, C, x_proj
        y = y_flat.view(B, S, M_OUT)
        # Use Triton to avoid torch.chunk
        B_tensor = y[:, :, :H]
        C_tensor = y[:, :, H:2*H]
        x_proj_tensor = y[:, :, 2*H:]

        # 2) Element-wise gating: Bx = B * x_proj
        Bx_flat = torch.empty((M, H), device=device, dtype=dtype)
        elementwise_mul_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 64))](
            B_tensor.reshape(M, H).contiguous(), x_proj_tensor.reshape(M, H).contiguous(),
            Bx_flat,
            M, H,
            B_tensor.reshape(M, H).stride(0), B_tensor.reshape(M, H).stride(1),
            x_proj_tensor.reshape(M, H).stride(0), x_proj_tensor.reshape(M, H).stride(1),
            Bx_flat.stride(0), Bx_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64,
        )

        # 3) Grouped causal 1D convolution: Out_conv[b, h, t] = sum_{k=0..3} Bx[b,h,t-k] * conv_weight[h,k] + conv_bias[h]
        # X_conv: (M=B*S, H, S)
        X_conv = Bx_flat.view(M, H, S).contiguous()
        W_conv = conv_weight  # (H, 4)
        Bias_conv = conv_bias  # (H,)
        Out_conv = torch.empty((M, H, S), device=device, dtype=dtype)

        grid_conv = (triton.cdiv(M, 128), triton.cdiv(H, 64), triton.cdiv(S, 128))
        grouped_causal_conv1d_kernel[grid_conv](
            X_conv, W_conv, Bias_conv, Out_conv,
            M, H, S, K,
            X_conv.stride(0), X_conv.stride(1), X_conv.stride(2),
            W_conv.stride(0), W_conv.stride(1),
            Out_conv.stride(0), Out_conv.stride(1), Out_conv.stride(2),
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=128,
        )

        # 4) Output gating: y = C * Out_conv
        C_flat = C_tensor.reshape(M, H).contiguous()
        y_gated_flat = torch.empty((M, H), device=device, dtype=dtype)
        elementwise_mul_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 64))](
            C_flat, Out_conv.reshape(M, H, S).reshape(M, H).contiguous(),
            y_gated_flat,
            M, H,
            C_flat.stride(0), C_flat.stride(1),
            Out_conv.reshape(M, H, S).reshape(M, H).stride(0), Out_conv.reshape(M, H, S).reshape(M, H).stride(1),
            y_gated_flat.stride(0), y_gated_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64,
        )

        # 5) Final projection: (B, S, H) = y_gated_flat.view(B, S, H) @ out_proj_weight^T + out_proj_bias
        # We need to treat (B*S, H) times (H, H)
        y_gated = y_gated_flat.view(M, H)
        out_proj_weight_t = out_proj_weight.transpose(0, 1).contiguous()  # (H, H)
        out_flat = torch.empty((M, H), device=device, dtype=dtype)

        grid_final = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        final_linear_gemm_bias_kernel[grid_final](
            y_gated, out_proj_weight_t, out_proj_bias, out_flat,
            M, H, H,
            y_gated.stride(0), y_gated.stride(1),
            out_proj_weight_t.stride(0), out_proj_weight_t.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
        )

        # Reshape back to (B, S, H)
        output = out_flat.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
