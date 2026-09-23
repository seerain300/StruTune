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

    # Loop over K dimension (H)
    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < H

        # Load X tile: shape [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xh)
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wm + k[:, None] * stride_wh)  # [BLOCK_N, BLOCK_K] then transpose
        # We need W as [BLOCK_K, BLOCK_N], so we load with k as rows and offs_n as cols
        w = tl.load(W_ptr + (offs_n[None, :] * stride_wm + k[:, None] * stride_wh), mask=k_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # acc += x @ w.T -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias[None, :]

    # Store output
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
    # We write three slices: B: first Bsz columns, C: next Csz columns, XPRJ: last Xsz columns, each [M, corresponding size]
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
    M, C_in, L, PAD, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
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
    # t_in = t + PAD - k, with left-pad PAD=3 and no right-pad
    for t0 in range(0, L, BLOCK_T):
        t = t0 + offs_t  # time indices for this tile
        ti_mask = t < L

        # Loop over kernel size K
        for k in range(0, K):
            t_in = t + PAD - k  # left-pad
            # Valid input positions where 0 <= t_in < L
            valid = (t_in >= 0) & (t_in < L) & ti_mask
            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for all c in offs_c
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]

            # Accumulate
            # Broadcast w_vals over rows
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias = tl.load(BIAS_ptr + offs_c, mask=c_mask, other=0.0)  # [BLOCK_C]
    acc = acc + bias[None, :]

    # Store output: Out[b, c, t]
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + t[None, :] * stride_oL)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & c_mask[None, :] & ti_mask[None, :])


@triton.jit
def final_proj_kernel(
    Y_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, H, M_out,
    stride_ym, stride_yn,
    stride_wm, stride_wh,
    stride_om, stride_om2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Y: [M, H] row-major, W: [M_out, H] row-major, Out: [M, M_out] row-major
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

        y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + k[None, :] * stride_yn)
        y = tl.load(y_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]

        w_ptrs = W_ptr + (offs_n[None, :] * stride_wm + k[:, None] * stride_wh)  # [BLOCK_K, BLOCK_N]
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(y, tl.trans(w))

    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_om2)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Shapes
        B, S, H = x.shape
        # 1) First linear projection: (B, S, H) -> (B, S, 3H)
        M = B * S
        X = x.reshape(M, H).contiguous()
        Y_flat = torch.empty((M, 3 * H), dtype=X.dtype, device=X.device)

        # Triton in-projection
        # Y_flat[i, j] = sum_h X[i, h] * in_proj_weight[j, h] + in_proj_bias[j]
        in_proj = in_proj_weight  # (M_out, H) with M_out=3H
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(3 * H, BLOCK_N))
        in_proj_linear_kernel[grid](
            X, in_proj, in_proj_bias, Y_flat,
            M, H, 3 * H,
            X.stride(0), X.stride(1),
            in_proj.stride(0), in_proj.stride(1),
            Y_flat.stride(0), Y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Chunk along dim=1 into B, C, x_proj
        B_sz = H
        C_sz = H
        X_sz = H
        Y = Y_flat.view(B, S, 3 * H).contiguous()  # (B, S, 3H), but we keep as flat for chunking along dim=1
        # We will re-read Y_flat directly for chunking along dim=1
        B_flat = torch.empty((M, H), dtype=Y_flat.dtype, device=Y_flat.device)
        C_flat = torch.empty((M, H), dtype=Y_flat.dtype, device=Y_flat.device)
        x_proj_flat = torch.empty((M, H), dtype=Y_flat.dtype, device=Y_flat.device)

        grid_chunk = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        chunk_dim1_3_kernel[grid_chunk](
            Y_flat, B_flat, C_flat, x_proj_flat,
            M, 3 * H, B_sz, C_sz, X_sz,
            Y_flat.stride(0), Y_flat.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            x_proj_flat.stride(0), x_proj_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Reshape back to (B, S, H)
        B_3d = B_flat.view(B, S, H)
        C_3d = C_flat.view(B, S, H)
        x_proj_3d = x_proj_flat.view(B, S, H)

        # 3) Element-wise gating
        Bx_flat = torch.empty((M, H), dtype=x.dtype, device=x.device)
        grid_mul = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        mul_elementwise_kernel[grid_mul](
            B_3d.reshape(M, H), x_proj_3d.reshape(M, H), Bx_flat,
            M, H,
            B_3d.reshape(M, H).stride(0), B_3d.reshape(M, H).stride(1),
            x_proj_3d.reshape(M, H).stride(0), x_proj_3d.reshape(M, H).stride(1),
            Bx_flat.stride(0), Bx_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        Bx = Bx_flat.view(B, S, H)

        # 4) Grouped causal conv1d in Triton
        # X for conv: (M, C_in, L) where C_in=H, L=S
        # We need left-pad 3: t_in = t + 3 - k; mask for 0 <= t_in < S
        conv_weight_t = conv_weight  # (C_in, K=4)
        conv_bias_t = conv_bias
        C_in = H
        L = S
        K = 4
        PAD = 3

        Bx_flat_conv = Bx.reshape(M, C_in, L).contiguous()  # (M, H, S)
        conv_out_flat = torch.empty((M, C_in, L), dtype=Bx.dtype, device=Bx.device)

        grid_conv = (triton.cdiv(M, 128), triton.cdiv(C_in, 64))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_flat_conv, conv_weight_t, conv_bias_t, conv_out_flat,
            M, C_in, L, PAD, K,
            Bx_flat_conv.stride(0), Bx_flat_conv.stride(1), Bx_flat_conv.stride(2),
            conv_weight_t.stride(0), conv_weight_t.stride(1),
            conv_out_flat.stride(0), conv_out_flat.stride(1), conv_out_flat.stride(2),
            BLOCK_M=128, BLOCK_C=64, BLOCK_T=128,
        )
        conv_out = conv_out_flat.view(B, H, S)

        # 5) Output gating: C * conv_out
        gated = C_3d * conv_out  # (B, H, S)

        # 6) Final output projection: (B, H, S) -> (B, S, H)
        Y_gate_flat = gated.reshape(B * S, H).contiguous()
        Out_flat = torch.empty((M, H), dtype=gated.dtype, device=gated.device)

        # Triton final projection: Y_gate_flat @ out_proj_weight^T + bias
        out_proj = out_proj_weight  # (H, H)
        out_bias = out_proj_bias
        grid_final = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        final_proj_kernel[grid_final](
            Y_gate_flat, out_proj, out_bias, Out_flat,
            M, H, H,
            Y_gate_flat.stride(0), Y_gate_flat.stride(1),
            out_proj.stride(0), out_proj.stride(1),
            Out_flat.stride(0), Out_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
        )

        output = Out_flat.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
