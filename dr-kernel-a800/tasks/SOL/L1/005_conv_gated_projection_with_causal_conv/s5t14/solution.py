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
        # Load W block: [BLOCK_N, BLOCK_K] (we need W[n, k])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # Accumulate: acc += x @ w^T => sum over k
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results
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
    # We write three slices: [:Bsz] -> B, [Bsz : Bsz+Csz] -> C, [Bsz+Csz : Bsz+Csz+Xsz] -> XPRJ, each [M, corresponding size]
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
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # For each output time position t, accumulate over kernel window
    # t_in = t - k (no padding, zero out of bounds)
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

        # Load W block: [BLOCK_N, BLOCK_K] (we need W[n, k])
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T => sum over k
        acc += tl.dot(x, tl.trans(w))

    # Add bias
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
        """
        Triton-orchestrated forward:
        1) in_proj_linear: y = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        2) chunk along dim=1: B, C, x_proj from y (B,S,3H)
        3) element-wise gating: Bx = B * x_proj
        4) grouped causal conv1d (groups=H, kernel_size=4, padding=0): conv_out = conv1d(Bx, conv_weight, conv_bias)
        5) output gating: y_g = C * conv_out
        6) final projection: out = F.linear(y_g, out_proj_weight, out_proj_bias)
        Note: No torch ops in host code except tensor creation and launches.
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        B, S, H = x.shape
        M = B * S
        M_out = 3 * H

        # 1) in_proj_linear: X_flat = x.view(M, H), W = in_proj_weight (M_out, H)
        X_flat = x.view(M, H).contiguous()
        y_flat = torch.empty((M, M_out), device=x.device, dtype=x.dtype)
        in_proj_linear_kernel[(triton.cdiv(M, 64), triton.cdiv(M_out, 64))](
            X_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_out,
            X_flat.stride(0), X_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 2) chunk along dim=1: B, C, x_proj from y_flat (shape (M, 3H))
        Bsz, Csz, Xsz = H, H, H
        Y = y_flat.view(B, S, 3 * H)
        B_flat = torch.empty((M, H), device=x.device, dtype=x.dtype)
        C_flat = torch.empty((M, H), device=x.device, dtype=x.dtype)
        XPRJ_flat = torch.empty((M, H), device=x.device, dtype=x.dtype)
        Y_view = y_flat.view(M, 3 * H)  # keep contiguous
        chunk_dim1_3_kernel[(triton.cdiv(M, 64), triton.cdiv(3 * H, 64))](
            Y_view, B_flat, C_flat, XPRJ_flat,
            M, 3 * H, Bsz, Csz, Xsz,
            Y_view.stride(0), Y_view.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            XPRJ_flat.stride(0), XPRJ_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64
        )
        B = B_flat.view(B, S, H)
        C = C_flat.view(B, S, H)
        x_proj = XPRJ_flat.view(B, S, H)

        # 3) element-wise gating
        Bx = torch.empty_like(B)
        mul_elementwise_kernel[(triton.cdiv(B.numel(), 64), triton.cdiv(H, 64))](
            B, x_proj, Bx,
            B.numel(), H,
            B.stride(0), B.stride(2),
            x_proj.stride(0), x_proj.stride(2),
            Bx.stride(0), Bx.stride(2),
            BLOCK_M=64, BLOCK_N=64
        )

        # 4) grouped causal conv1d: groups=H, kernel_size=4, padding=0
        # X_conv: [M, H, S] => we pass Bx view as (M, H, S). conv_weight: [H, 4]
        X_conv = Bx.contiguous().view(M, H, S)
        W_conv = conv_weight.contiguous()  # [H, 4]
        Bias_conv = conv_bias.contiguous()  # [H]
        Out_conv = torch.empty((M, H, S), device=x.device, dtype=x.dtype)
        grouped_causal_conv1d_kernel[(triton.cdiv(M, 64), triton.cdiv(H, 64))](
            X_conv, W_conv, Bias_conv, Out_conv,
            M, H, S, 4,
            X_conv.stride(0), X_conv.stride(1), X_conv.stride(2),
            W_conv.stride(0), W_conv.stride(1),
            Out_conv.stride(0), Out_conv.stride(1), Out_conv.stride(2),
            BLOCK_M=64, BLOCK_C=64, BLOCK_T=64
        )
        # Reshape back to (B, H, S)
        conv_out = Out_conv.view(B, H, S)

        # 5) output gating: y_g = C * conv_out
        y_g = C * conv_out  # shape (B, H, S)

        # 6) final projection: y_g -> (B, S, H)
        y_g_flat = y_g.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), device=x.device, dtype=x.dtype)
        final_proj_kernel[(triton.cdiv(M, 64), triton.cdiv(H, 64))](
            y_g_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H, H,
            y_g_flat.stride(0), y_g_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )
        out = out_flat.view(B, S, H)
        return out


def run(*args):
    return ModelNew()(*args)
