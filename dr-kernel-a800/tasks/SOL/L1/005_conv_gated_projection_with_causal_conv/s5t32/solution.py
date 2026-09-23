import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X: [M, H] where M = B*S, W: [M_OUT, H], OUT: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    k_mask = offs_k < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_in_k_mask = k < H
        # Load X block [BLOCK_M, BLOCK_K] using broadcast pointers
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_in_k_mask[None, :], other=0.0)
        # Load W block [BLOCK_K, BLOCK_K] (note: we want W[k, :])
        w_ptrs = W_ptr + (k[:, None] * stride_wm + offs_k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=k_in_k_mask[:, None] & k_mask[None, :], other=0.0)
        # Accumulate: acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias: BIAS[k] per output channel
    bias = tl.load(BIAS_ptr + offs_k, mask=k_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_oh)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & k_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    IN_ptr, OUT1_ptr, OUT2_ptr, OUT3_ptr,
    M, C_IN,
    stride_im, stride_in,
    stride_o1m, stride_o1n,
    stride_o2m, stride_o2n,
    stride_o3m, stride_o3n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, 3*C_IN] where M = B*S
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over C_IN (each output corresponds to one of 3 chunks)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_IN

    # For chunk 1 (first C_IN): base = 0
    in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] * 0) * stride_in)
    out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_o1m + offs_n[None, :] * stride_o1n)
    val1 = tl.load(in_ptrs1, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out1_ptrs, val1, mask=m_mask[:, None] & n_mask[None, :])

    # For chunk 2 (middle C_IN): base = C_IN
    in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] * C_IN) * stride_in)
    out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_o2m + offs_n[None, :] * stride_o2n)
    val2 = tl.load(in_ptrs2, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out2_ptrs, val2, mask=m_mask[:, None] & n_mask[None, :])

    # For chunk 3 (last C_IN): base = 2*C_IN
    in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (offs_n[None, :] * (2 * C_IN)) * stride_in)
    out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_o3m + offs_n[None, :] * stride_o3n)
    val3 = tl.load(in_ptrs3, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(out3_ptrs, val3, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, H,
    stride_bm, stride_bh,
    stride_xm, stride_xh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # B: [M, H], XPRJ: [M, H], OUT: [M, H], where M = B*S
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    h_mask = offs_h < H

    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_h[None, :] * stride_bh)
    x_ptrs = XPRJ_ptr + (offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh)
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh)

    b = tl.load(b_ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    x = tl.load(x_ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    y = b * x
    tl.store(out_ptrs, y, mask=m_mask[:, None] & h_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_in, L] (we pass M=B*S, C_in=H, L=S)
    # W_ptr: [C_in, K]
    # Out_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, BLOCK_T)

    m_mask = offs_m < M
    c_mask = offs_c < C_in
    t_mask = offs_t < L

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
        k = k0 + offs_k
        k_mask = k < IN_H
        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W block: [BLOCK_N, BLOCK_K] (note: W[n, k] so we index by n and k)
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # Accumulate: acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias: BIAS[n] per output channel
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # x: (B, S, H)
        B, S, H = x.shape
        C_in = H  # since y is split into 3 chunks of size H
        M = B * S

        # 1) First linear: in_proj
        # Flatten x to (M, H)
        x_flat = x.reshape(M, H)
        y_flat = torch.empty((M, 3 * H), dtype=x.dtype, device=x.device)
        # Strides: X is [M, H], y is [M, 3H]
        in_proj_linear_kernel[(triton.cdiv(M, 128), triton.cdiv(3 * H, 128))](  # grid for demonstration; adjust if needed
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, 3 * H,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=128, BLOCK_K=128
        )
        # Reshape back to (B, S, 3H)
        y = y_flat.reshape(B, S, 3 * H)

        # 2) Split y along dim=1 into B, C, x_proj (each shape (B, S, H))
        B_tensor = y[:, :, :H]
        C_tensor = y[:, :, H:2 * H]
        x_proj = y[:, :, 2 * H:]

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty_like(B_tensor)
        mul_elementwise_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 128))](  # grid for demonstration; adjust if needed
            B_tensor.reshape(M, H), x_proj.reshape(M, H), Bx.reshape(M, H),
            M, H,
            B_tensor.reshape(M, H).stride(0), B_tensor.reshape(M, H).stride(1),
            x_proj.reshape(M, H).stride(0), x_proj.reshape(M, H).stride(1),
            Bx.reshape(M, H).stride(0), Bx.reshape(M, H).stride(1),
            BLOCK_M=128, BLOCK_H=128
        )
        Bx = Bx.reshape(B, S, H)

        # 4) Grouped causal 1D convolution on Bx with kernel_size=4, groups=H (depthwise), padding=0
        # X shape for conv1d: (M=B*S, C_in=H, L=S)
        X_for_conv = Bx.transpose(1, 2).contiguous()  # (B*S, H, S)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)

        # Strides for Triton kernel: X_for_conv strides (M, C, L)
        stride_xM = X_for_conv.stride(0)  # H*S
        stride_xC = X_for_conv.stride(1)  # S
        stride_xL = X_for_conv.stride(2)  # 1

        # conv_weight: (C_in=H, K=4), conv_bias: (C_in=H)
        stride_wC = conv_weight.stride(0)  # 1
        stride_wK = conv_weight.stride(1)  # 1

        # Output conv_out: (M=B*S, C_in=H, L=S)
        stride_oM = conv_out.stride(0)  # H
        stride_oC = conv_out.stride(1)  # S
        stride_oL = conv_out.stride(2)  # 1

        grouped_causal_conv1d_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 128))](
            X_for_conv, conv_weight, conv_bias, conv_out,
            M, H, S, 4,
            stride_xM, stride_xC, stride_xL,
            stride_wC, stride_wK,
            stride_oM, stride_oC, stride_oL,
            BLOCK_M=128, BLOCK_C=128, BLOCK_T=128
        )

        # 5) Output gating: y = C * conv_out (shape (B, H, S))
        y_out = C_tensor * conv_out  # broadcasting

        # 6) Final projection: y_out (B,S,H) -> out (B,S,H)
        y_out_flat = y_out.reshape(M, H)
        out_flat = torch.empty((M, H), dtype=x.dtype, device=x.device)
        final_proj_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 128))](  # grid for demonstration; adjust if needed
            y_out_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H, H,
            y_out_flat.stride(0), y_out_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128
        )
        out = out_flat.reshape(B, S, H)

        return out


def run(*args):
    return ModelNew()(*args)
