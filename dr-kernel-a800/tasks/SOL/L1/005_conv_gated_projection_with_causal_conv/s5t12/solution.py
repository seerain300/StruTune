import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, H, M_out,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_om2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X: [M, H] row-major, W: [M_out, H] row-major, Out: [M, M_out] row-major
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < M_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < H
        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # Accumulate: acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias[None, :]

    # Store
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_om2)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    Y_ptr, B_ptr, C_ptr, XPRJ_ptr,
    M, YN, Bsz, Csz, Xsz,
    stride_ym, stride_yn,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Y_ptr points to a tensor of shape [M, YN] (YN = 3*H here).
    # We write three slices: B_flat: first Bsz columns -> [M, Bsz]
    #                       C_flat: next Csz columns -> [M, Csz]
    #                       XPRJ_flat: last Xsz columns -> [M, Xsz]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M

    # B = first Bsz columns
    base = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    b_vals = tl.load(Y_ptr + base, mask=m_mask[:, None] & (offs_n < Bsz)[None, :], other=0.0)
    tl.store(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, b_vals, mask=m_mask[:, None] & (offs_n < Bsz)[None, :])

    # C = next Csz columns
    c_base = offs_m[:, None] * stride_ym + (Bsz + offs_n[None, :]) * stride_yn
    c_vals = tl.load(Y_ptr + c_base, mask=m_mask[:, None] & (offs_n < Csz)[None, :], other=0.0)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_vals, mask=m_mask[:, None] & (offs_n < Csz)[None, :])

    # XPRJ = last Xsz columns
    xp_base = offs_m[:, None] * stride_ym + (Bsz + Csz + offs_n[None, :]) * stride_yn
    xp_vals = tl.load(Y_ptr + xp_base, mask=m_mask[:, None] & (offs_n < Xsz)[None, :], other=0.0)
    tl.store(XPRJ_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, xp_vals, mask=m_mask[:, None] & (offs_n < Xsz)[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, H,
    stride_bm, stride_bn,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # B: [M, H], XPRJ: [M, H], OUT: [M, H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x = tl.load(XPRJ_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = b * x
    tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=m_mask[:, None] & n_mask[None, :])


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
        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # Accumulate: acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Shapes
        Bsz, S, H = x.shape
        M = Bsz * S
        M_out = in_proj_weight.shape[0]  # 3*H
        C_in = conv_weight.shape[0]      # H
        K = conv_weight.shape[2]         # 4

        # 1) in-projection linear: y = F.linear(x, in_proj_weight, in_proj_bias)
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, M_out), device=x.device, dtype=torch.float32)
        # Grid: (ceil_div(M, BLOCK_M), ceil_div(M_out, BLOCK_N))
        in_proj_linear_kernel[(triton.cdiv(M, 64), triton.cdiv(M_out, 64))](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4,
        )
        y_flat = y_flat  # dtype float32 for numerical stability

        # 2) chunk y_flat along dim=1 (channels) into B, C, x_proj, each shape (M, H)
        y = y_flat.view(Bsz, S, M_out)  # y: (B, S, 3H)
        # For Triton, flatten (B, S) -> M and (3H) -> YN
        y_flat2 = y.reshape(M, M_out).contiguous()
        B_flat = torch.empty((M, H), device=x.device, dtype=torch.float32)
        C_flat = torch.empty((M, H), device=x.device, dtype=torch.float32)
        x_proj_flat = torch.empty((M, H), device=x.device, dtype=torch.float32)
        # Grid over M and H
        chunk_dim1_3_kernel[(triton.cdiv(M, 128), triton.cdiv(M_out // 3, 128))](
            y_flat2, B_flat, C_flat, x_proj_flat,
            M, M_out, H, H, H,
            y_flat2.stride(0), y_flat2.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            x_proj_flat.stride(0), x_proj_flat.stride(1),
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4,
        )

        B = B_flat.view(Bsz, S, H)
        C = C_flat.view(Bsz, S, H)
        x_proj = x_proj_flat.view(Bsz, S, H)

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((Bsz, S, H), device=x.device, dtype=torch.float32)
        mul_elementwise_kernel[(triton.cdiv(Bsz * S, 128), triton.cdiv(H, 128))](
            B, x_proj, Bx,
            Bsz * S, H,
            B.stride(0), B.stride(2),
            x_proj.stride(0), x_proj.stride(2),
            Bx.stride(0), Bx.stride(2),
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4,
        )

        # 4) Grouped causal conv1d: conv_out = conv(Bx, conv_weight, conv_bias, groups=H, kernel_size=4)
        # X: (M=B*S, C_in=H, L=S), W: (C_in=H, K=4), Bias: (H,)
        Bx_flat = Bx.reshape(M, H).contiguous()
        conv_out = torch.empty((M, C_in), device=x.device, dtype=torch.float32)
        grouped_causal_conv1d_kernel[(triton.cdiv(M, 128), triton.cdiv(C_in, 32))](
            Bx_flat, conv_weight, conv_bias, conv_out,
            M, C_in, S, K,
            Bx_flat.stride(0), Bx_flat.stride(1), Bx_flat.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=128, BLOCK_C=32, BLOCK_T=128,
            num_warps=4,
        )
        conv_out = conv_out.view(Bsz, S, C_in)  # (B, S, H)

        # 5) Output gating: y = C * conv_out
        gated = C * conv_out  # (B, S, H)

        # 6) Final projection: y @ out_proj_weight^T + out_proj_bias
        gated_flat = gated.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), device=x.device, dtype=torch.float32)
        final_proj_kernel[(triton.cdiv(M, 128), triton.cdiv(H, 128))](
            gated_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H, H,
            gated_flat.stride(0), gated_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4,
        )

        # Reshape back to (B, S, H)
        out = out_flat.view(Bsz, S, H)

        return out


def run(*args):
    return ModelNew()(*args)
