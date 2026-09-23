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
    # X: [M, H], W: [M_out, H], Out: [M, M_out]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < M_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        a = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            W_ptr + offs_n[None, :] * stride_wm + k_ids[:, None] * stride_wh,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(b))

    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_om2,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def chunk3_along_dim1_kernel(
    Y_ptr, B_ptr, C_ptr, XPRJ_ptr,
    M, YN, Bsz, Csz, Xsz,
    stride_ym, stride_yn,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Y_ptr points to a tensor of shape [M, YN] where YN must equal Bsz + Csz + Xsz.
    # We write three slices along dim=1: [:Bsz] -> B, [Bsz : Bsz+Csz] -> C, [Bsz+Csz : Bsz+Csz+Xsz] -> XPRJ.
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
def pad_left_kernel(
    IN_ptr, OUT_ptr,
    M, L_in, PAD,
    stride_im, stride_in,
    stride_om, stride_on,
):
    # IN: [M, L_in], OUT: [M, L_in + PAD] (only left pad, right remains unchanged)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * M + tl.arange(0, M)
    offs_n = pid_n * (L_in + PAD) + tl.arange(0, (L_in + PAD))
    m_mask = offs_m < M
    n_mask = offs_n < (L_in + PAD)

    # For n < PAD, write 0
    is_pad = offs_n < PAD
    # For n >= PAD, write IN[m, n - PAD]
    in_n = offs_n - PAD
    in_mask = m_mask & (~is_pad) & (in_n >= 0) & (in_n < L_in)

    in_vals = tl.load(IN_ptr + offs_m[:, None] * stride_im + in_n[None, :] * stride_in, mask=in_mask[:, None], other=0.0)
    tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, in_vals, mask=m_mask[:, None] & (~is_pad)[None, :])
    # Pad region remains zero due to using masked load with other=0.0


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, PAD, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_in, L] (we pass M=B*S, C_in=H, L=S)
    # W_ptr: [C_in, K]
    # Out_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    m = pid_m
    c = pid_c

    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for k in range(0, K):
        t_in = offs_t + PAD - k  # left pad: k in [0..K-1]
        valid = (t_in >= 0) & (t_in < L) & t_mask
        x_idx = m * stride_xM + c * stride_xC + t_in * stride_xL
        x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
        w_val = tl.load(W_ptr + c * stride_wC + k * stride_wK)
        acc += x_val * w_val

    bias = tl.load(BIAS_ptr + c)
    acc += bias

    out_idx = m * stride_oM + c * stride_oC + offs_t * stride_oL
    tl.store(Out_ptr + out_idx, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X: [M, N], W: [N, N] (since out_proj_weight is (H, H)), Out: [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < N

        a = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xn,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            W_ptr + k_ids[:, None] * stride_wm + offs_n[None, :] * stride_wn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


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
        x: (B, S, H)
        in_proj_weight: (M_out, H), M_out=3*H
        in_proj_bias: (M_out,)
        conv_weight: (H, K=4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        device = x.device
        B, S, H = x.shape
        M_out = in_proj_weight.shape[0]

        # 1) In-projection: y_flat = x_flat @ W^T + bias
        x_flat = x.reshape(B * S, H).contiguous()  # (M=B*S, H)
        y_flat = torch.empty((B * S, M_out), dtype=x.dtype, device=device)  # (M, M_out)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid_in = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            B * S, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Chunk along dim=1 of (B, S, 3H): get B, C, x_proj each (B, S, H)
        # y_flat is (M=B*S, 3H). We chunk along dim=1 (columns).
        YN = 3 * H
        Bsz, Csz, Xsz = H, H, H
        B_flat = torch.empty((B * S, Bsz), dtype=x.dtype, device=device)  # (M, H)
        C_flat = torch.empty((B * S, Csz), dtype=x.dtype, device=device)  # (M, H)
        x_proj_flat = torch.empty((B * S, Xsz), dtype=x.dtype, device=device)  # (M, H)

        # Launch chunk3 along dim=1: YN=3H, Bsz=Csz=Xsz=H
        grid_chunk = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(3 * H, BLOCK_N))
        chunk3_along_dim1_kernel[grid_chunk](
            y_flat, B_flat, C_flat, x_proj_flat,
            B * S, YN, Bsz, Csz, Xsz,
            y_flat.stride(0), y_flat.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            x_proj_flat.stride(0), x_proj_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Reshape to (B, S, H)
        B_part = B_flat.view(B, S, H)
        C_part = C_flat.view(B, S, H)
        x_proj = x_proj_flat.view(B, S, H)

        # 3) Element-wise gating: Bx = B_part * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=device)
        grid_mul = (B, S, H)
        mul_elementwise_kernel[grid_mul](
            B_part, x_proj, Bx,
            B * S, H,
            B_part.stride(0), B_part.stride(1),
            x_proj.stride(0), x_proj.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        # 4) Pre-pad Bx on left by 3 for causal conv
        Bx_reshaped = Bx.reshape(B * S, H).contiguous()  # (M, H)
        L_in = H
        PAD = 3
        L_out = L_in + PAD  # padded length
        Bx_padded = torch.empty((B * S, L_out), dtype=x.dtype, device=device)  # (M, L_out)
        grid_pad = (B * S, triton.cdiv(L_out, 256))
        pad_left_kernel[grid_pad](
            Bx_reshaped, Bx_padded,
            B * S, L_in, PAD,
            Bx_reshaped.stride(0), Bx_reshaped.stride(1),
            Bx_padded.stride(0), Bx_padded.stride(1),
            num_warps=4, num_stages=2
        )

        # 5) Grouped causal conv1d: groups=H, K=4, PAD=3
        conv_out_flat = torch.empty((B * S, H), dtype=x.dtype, device=device)  # (M, H)
        grid_conv = (B * S, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out_flat,
            B * S, H, S, PAD, 4,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out_flat.stride(0), conv_out_flat.stride(1), conv_out_flat.stride(2),
            BLOCK_C=1, BLOCK_T=256,
            num_warps=4, num_stages=2
        )

        conv_out = conv_out_flat.view(B, S, H).contiguous()

        # 6) Output gating: y = C_part * conv_out
        y_gate = C_part * conv_out  # (B, S, H)

        # 7) Final projection: y_gate @ out_proj_weight^T + out_proj_bias
        M = B * S
        y_gate_flat = y_gate.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), dtype=x.dtype, device=device)

        BLOCK_M_out, BLOCK_N_out, BLOCK_K_out = 64, 64, 32
        grid_out = (triton.cdiv(M, BLOCK_M_out), triton.cdiv(H, BLOCK_N_out))
        out_proj_linear_kernel[grid_out](
            y_gate_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H,
            y_gate_flat.stride(0), y_gate_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=BLOCK_M_out, BLOCK_N=BLOCK_N_out, BLOCK_K=BLOCK_K_out,
            num_warps=4, num_stages=2
        )

        output = out_flat.view(B, S, H).contiguous()
        return output


def run(*args):
    return ModelNew()(*args)
