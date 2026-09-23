import torch
import triton
import triton.language as tl

@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Compute Y = X @ W^T + bias
    X: [M, IN_H] float32
    W: [OUT_H, IN_H] float32
    OUT: [M, OUT_H] float32
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension (IN_H)
    for k0 in range(0, IN_H, 1):
        # Load X row block: [BLOCK_M]
        x_ptrs = X_ptr + offs_m * stride_xm + k0 * stride_xn
        x_vals = tl.load(x_ptrs, mask=m_mask, other=0.0)  # [BLOCK_M]

        # Load W column block: [BLOCK_N]
        w_ptrs = W_ptr + offs_n * stride_wm + k0 * stride_wn
        w_vals = tl.load(w_ptrs, mask=n_mask, other=0.0)  # [BLOCK_N]

        # Outer product: x_vals[:, None] * w_vals[None, :]
        acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    A_ptr, B_ptr, OUT_ptr,
    M, N,
    stride_am, stride_an, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    OUT = A * B
    A: [M, N], B: [M, N]
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=m_mask[:, None], other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, mask=m_mask[:, None], other=0.0)
    out = a * b
    tl.store(OUT_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, out, mask=m_mask[:, None])


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
      Weight W: [C_in, K]
      Output Out: [M, C_in, L]
      groups = C_in: per-channel depthwise conv
      No padding (conv1d default), kernel_size = K (assumed 4 here).
    """
    pid_m = tl.program_id(0)  # over M (batch*seq)
    pid_c = tl.program_id(1)  # over output channels (C_in)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    # Accumulator for current output channels BLOCK_C across BLOCK_M rows
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Loop over output positions t in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        ti_mask = t < L

        # For each kernel position k, compute input time t_in = t - k
        for k in range(0, K):
            t_in = t - k  # causal: k in {0,1,2,3} => t_in in [t-3, t]
            valid = (t_in >= 0) & (t_in < L) & ti_mask  # zero padding when t - k < 0

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
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + tl.arange(0, BLOCK_T))[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + tl.arange(0, BLOCK_T))[None, :] < L
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
    Compute OUT = IN @ W^T + bias
    IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
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
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W block as [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + (k[:, None] * stride_wm + offs_n[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += x @ w
        acc += tl.dot(x, w)

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
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
        in_proj_bias_ = in_proj_bias.contiguous()
        in_proj_weight_ = in_proj_weight.contiguous()

        grid = (triton.cdiv(M, 128), triton.cdiv(3 * H, 64))
        in_proj_linear_kernel[grid](
            x_flat, in_proj_weight_, in_proj_bias_, in_proj_out,
            M, H, 3 * H,
            1, 1,  # strides for X: row-major (M, H) -> stride_xm=H, stride_xn=1
            in_proj_weight_.stride(0), in_proj_weight_.stride(1),
            in_proj_out.stride(0), in_proj_out.stride(1),
            BLOCK_M=128, BLOCK_N=64,
        )

        # 2) Split in_proj_out (M, 3H) into B, C, x_proj along channels (size H):
        #   We assume in_proj_out is (M, 3H) and split via indexing:
        #   B: first H columns, C: middle H columns, x_proj: last H columns
        Bt = in_proj_out[:, :H].contiguous().reshape(B, S, H)
        Ct = in_proj_out[:, H:2 * H].contiguous().reshape(B, S, H)
        xprj = in_proj_out[:, 2 * H:].contiguous().reshape(B, S, H)

        # 3) Element-wise gating
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        # Launch Triton kernel: A=Bt, B=xprj, OUT=Bx
        grid2 = (triton.cdiv(B * S, 128), triton.cdiv(H, 64))
        mul_elementwise_kernel[grid2](
            Bt.reshape(B * S, H), xprj.reshape(B * S, H),
            Bx.reshape(B * S, H),
            B * S, H,
            1, 1, 1, 1,
            BLOCK_M=128, BLOCK_N=64,
        )

        # 4) Grouped causal 1D convolution (depthwise per channel) on Bx
        # Input X for conv: (M=B*S, C_in=H, L=S), contiguous
        X_conv = Bx.reshape(M, H, S).contiguous()
        Out_conv = torch.empty((M, H, S), dtype=x.dtype, device=x.device)

        # Weight conv: (C_in=H, K=4)
        conv_weight_ = conv_weight.contiguous()
        conv_bias_ = conv_bias.contiguous()

        # Launch grouped causal conv1d Triton kernel
        grid3 = (triton.cdiv(M, 128), triton.cdiv(H, 64), triton.cdiv(S, 64))
        grouped_causal_conv1d_kernel[grid3](
            X_conv, conv_weight_, conv_bias_, Out_conv,
            M, H, S, 4,
            1, 1, 1,  # X strides for (M, C_in, L)
            conv_weight_.stride(0), conv_weight_.stride(1),
            Out_conv.stride(0), Out_conv.stride(1), Out_conv.stride(2),
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=64,
        )

        # 5) Output gating: y = C * conv_out, conv_out is Out_conv (M, H, S)
        # We need to align C to shape (M, H, S). Here C is (B, S, H); we broadcast along L dim.
        C_ = Ct.reshape(B, S, H)
        y_gated = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        # For simplicity, we can compute element-wise y_gated[b,h,:] = C_[b,:,h] * Out_conv[b,h,:].
        # We'll launch a small Triton kernel to do this.
        for b in range(B):
            # Reshape to (M, H, S) slice for fixed b
            C_b = C_[b].reshape(1, H, S).contiguous()  # (1, H, S)
            Out_b = Out_conv[b].reshape(1, H, S).contiguous()  # (1, H, S)
            y_b = torch.empty((1, H, S), dtype=x.dtype, device=x.device)
            grid4 = (triton.cdiv(1, 1), triton.cdiv(H, 64), triton.cdiv(S, 64))
            mul_elementwise_kernel[grid4](
                C_b.reshape(1, H * S), Out_b.reshape(1, H * S),
                y_b.reshape(1, H * S),
                1, H * S,
                1, 1, 1, 1,
                BLOCK_M=1, BLOCK_N=64,
            )
            y_gated[b] = y_b[0]

        # 6) Final projection: y_gated @ out_proj_weight^T + out_proj_bias
        # y_gated: (B, H, S), flatten to (M, H)
        y_flat = y_gated.reshape(M, H).contiguous()
        out_proj_out = torch.empty((M, H), dtype=x.dtype, device=x.device)
        out_proj_weight_ = out_proj_weight.contiguous()
        out_proj_bias_ = out_proj_bias.contiguous()

        grid6 = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        final_proj_kernel[grid6](
            y_flat, out_proj_weight_, out_proj_bias_, out_proj_out,
            M, H, H,
            1, 1,
            out_proj_weight_.stride(0), out_proj_weight_.stride(1),
            out_proj_out.stride(0), out_proj_out.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
        )

        # Reshape back to (B, S, H)
        out = out_proj_out.reshape(B, S, H).contiguous()
        return out


def run(*args):
    return ModelNew()(*args)
