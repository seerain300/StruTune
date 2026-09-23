import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, C_IN, C_OUT,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X: [M, C_IN], W: [C_OUT, C_IN], OUT: [M, C_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C_IN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < C_IN

        # Load X block: [BM, BK]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BN, BK]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wn)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        for kk in range(BLOCK_K):
            if k0 + kk < C_IN:
                x_col = x[:, kk]  # [BM]
                w_col = w[:, kk]  # [BN]
                acc += x_col[:, None] * w_col[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    IN_ptr, OUT1_ptr, OUT2_ptr, OUT3_ptr,
    M, C,  # M = B * S, C = H
    stride_im, stride_in,  # IN is (M, 3*C)
    stride_ob1m, stride_ob1n,  # OUT1 (M, C)
    stride_ob2m, stride_ob2n,  # OUT2 (M, C)
    stride_ob3m, stride_ob3n,  # OUT3 (M, C)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, 3*C], OUTi: [M, C]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C

    # Iterate over channels and write to three outputs corresponding to channels 0..C-1, C..2C-1, 2C..3C-1
    for n0 in range(0, C, BLOCK_N):
        nn = n0 + offs_n
        nn_mask = nn < C

        # Output 1: channel 0..C-1
        in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + nn[None, :] * stride_in)
        vals1 = tl.load(in_ptrs1, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_ob1m + nn[None, :] * stride_ob1n)
        tl.store(out1_ptrs, vals1, mask=m_mask[:, None] & nn_mask[None, :])

        # Output 2: channel C..2C-1
        in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (C + nn[None, :]) * stride_in)
        vals2 = tl.load(in_ptrs2, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_ob2m + nn[None, :] * stride_ob2n)
        tl.store(out2_ptrs, vals2, mask=m_mask[:, None] & nn_mask[None, :])

        # Output 3: channel 2C..3C-1
        in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (2 * C + nn[None, :]) * stride_in)
        vals3 = tl.load(in_ptrs3, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_ob3m + nn[None, :] * stride_ob3n)
        tl.store(out3_ptrs, vals3, mask=m_mask[:, None] & nn_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, C,  # M = B*S, C = H
    stride_bm, stride_bn,  # B is (M, C)
    stride_xm, stride_xn,  # x_proj is (M, C)
    stride_om, stride_on,  # OUT is (M, C)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C

    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    x_ptrs = XPRJ_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    b = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = b * x

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_IN, L, PAD, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_IN, L] (we pass M=B*S, C_in=H, L=S)
    # W_ptr: [C_IN, K]
    # Out_ptr: [M, C_IN, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_IN
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # For each output time position t, accumulate over kernel window
    # t_in = t - k (no padding, zero out of bounds due to mask)
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        for k in range(0, K):
            t_in = t - k  # causal: k in {0,1,2,3} => t_in in [t-3, t]
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c  # bias[c_in] per output channel
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to Out[b, c, t]
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + offs_t)[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + offs_t)[None, :] < L
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
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IN_H

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        for kk in range(BLOCK_K):
            if k0 + kk < IN_H:
                x_col = x[:, kk]  # [BM]
                w_col = w[:, kk]  # [BN]
                acc += x_col[:, None] * w_col[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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
        # Ensure tensors are contiguous and on the same device
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        M = B * S

        # 1) First linear: y = F.linear(x, in_proj_weight, in_proj_bias)
        # in_proj_weight: (M_out, H), M_out = 3*H
        M_out = in_proj_weight.shape[0]
        X_flat = x.view(M, H)  # (M, H)
        y = torch.empty((M, M_out), device=x.device, dtype=x.dtype)

        # Launch in_proj_linear_kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        in_proj_linear_kernel[grid](
            X_flat, in_proj_weight, in_proj_bias, y,
            M, H, M_out,
            X_flat.stride(0), X_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Reshape y to (B, S, 3H)
        y = y.view(B, S, 3 * H)

        # 2) Split y along dim=1 (channels): B = y[:, :, :H], C = y[:, :, H:2H], x_proj = y[:, :, 2H:3H]
        B_tensor = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        C_tensor = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        x_proj = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        # Launch chunk_dim1_3_kernel
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        chunk_dim1_3_kernel[grid](
            y, B_tensor, C_tensor, x_proj,
            M, H,
            y.stride(0), y.stride(2),  # y is (B, S, 3H) with stride(2) for channels
            B_tensor.stride(0), B_tensor.stride(2),
            C_tensor.stride(0), C_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        mul_elementwise_kernel[grid](
            B_tensor, x_proj, Bx,
            M, H,
            B_tensor.stride(0), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(2),
            Bx.stride(0), Bx.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 4) Grouped causal 1D convolution: conv with groups=H, kernel_size=4, left-pad=3
        # X: (M=B*S, C_in=H, L=S), W: (C_in=H, K=4), bias: (C_in=H)
        X_conv = Bx.contiguous()  # (B, S, H)
        # We need X_conv reshaped as (M, C_in, L) -> (B*S, H, S)
        X_conv_flat = X_conv.view(M, H, S)
        conv_weight_t = conv_weight.transpose(0, 1).contiguous()  # (K, C_in) not used directly
        conv_bias_t = conv_bias.contiguous()

        # Allocate output conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)

        # Launch grouped_causal_conv1d_kernel
        BLOCK_M = 128
        BLOCK_C = 64
        BLOCK_T = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_C))
        grouped_causal_conv1d_kernel[grid](
            X_conv_flat, conv_bias_t, conv_out,
            M, H, S, 3, 4,  # PAD=3, K=4
            X_conv_flat.stride(0), X_conv_flat.stride(1), X_conv_flat.stride(2),
            conv_bias_t.stride(0), 1,  # stride_wC = 1 for (C_in,), stride_wK = 1 for K
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T,
        )

        # 5) Output gating: y = C * conv_out, shape (B, H, S)
        y_gated = C_tensor * conv_out  # broadcast multiply

        # 6) Final projection: y_gated @ out_proj_weight^T + out_proj_bias
        # y_gated: (B, H, S) -> (M=B*S, H)
        y_flat = y_gated.reshape(M, H)
        out = torch.empty((M, H), device=x.device, dtype=x.dtype)

        # Launch final_proj_kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        final_proj_kernel[grid](
            y_flat, out_proj_weight, out_proj_bias, out,
            M, H, H,
            y_flat.stride(0), y_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Reshape to (B, S, H)
        out = out.view(B, S, H)
        return out


def run(*args):
    return ModelNew()(*args)
