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

    # accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        # Load X tile: shape [BLOCK_M, BLOCK_K]
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load W tile: shape [BLOCK_N, BLOCK_K] (we need W^T here)
        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wm + k_ids[:, None] * stride_wh,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )

        # acc += x @ w.T
        acc += tl.dot(x, tl.trans(w))

    # add bias: shape [BLOCK_N]
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # store results
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_om2,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def chunk3_kernel(
    Y_ptr, Bsz, Csz, xps, OutB_ptr, OutC_ptr, OutX_ptr,
    M,  # M = B * S
    stride_yM, stride_yN,
    stride_bM, stride_bN,
    stride_cM, stride_cN,
    stride_xM, stride_xN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Y_ptr points to [M, 3H], OutB/OutC/OutX pointers point to [M, H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < xps  # xps is the size of x_proj (H)

    # For B: first Bsz columns
    # For C: next Csz columns
    # For x_proj: last xps columns (xps=Bsz here, but we pass H explicitly)
    for which in range(3):
        start = tl.where(which == 0, 0, tl.where(which == 1, Bsz, 2 * Bsz))
        # load from Y: columns = start + n
        y_cols = start + offs_n
        vals = tl.load(
            Y_ptr + offs_m[:, None] * stride_yM + y_cols[None, :] * stride_yN,
            mask=m_mask[:, None] & (y_cols[None, :] < (Bsz + Csz + xps)),
            other=0.0,
        )
        if which == 0:
            tl.store(OutB_ptr + offs_m[:, None] * stride_bM + offs_n[None, :] * stride_bN, vals, mask=m_mask[:, None] & n_mask[None, :])
        elif which == 1:
            tl.store(OutC_ptr + offs_m[:, None] * stride_cM + offs_n[None, :] * stride_cN, vals, mask=m_mask[:, None] & n_mask[None, :])
        else:
            tl.store(OutX_ptr + offs_m[:, None] * stride_xM + offs_n[None, :] * stride_xN, vals, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def elementwise_mul_kernel(
    B_ptr, X_ptr, Out_ptr,
    M, H,
    stride_bM, stride_bh,
    stride_xM, stride_xh,
    stride_oM, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Inputs B: [M, H], X: [M, H], Output Out: [M, H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    b = tl.load(B_ptr + offs_m[:, None] * stride_bM + offs_n[None, :] * stride_bh, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x = tl.load(X_ptr + offs_m[:, None] * stride_xM + offs_n[None, :] * stride_xh, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = b * x
    tl.store(Out_ptr + offs_m[:, None] * stride_oM + offs_n[None, :] * stride_oh, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, PAD, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr points to [M, C_in, L]; Out_ptr points to [M, C_in, L]; W_ptr points to [C_in, K]
    # Here we treat N=M and groups=C_in. For each (m, c), conv along L with K and left pad PAD.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    m = pid_m
    c = pid_c

    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < L

    # accumulator for this (m, c)
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # reduction over kernel size K
    for k in range(0, K):
        t_in = offs_t + PAD - k  # left pad, k in [0..K-1]
        valid = (t_in >= 0) & (t_in < L) & t_mask
        # index into X: m * stride_xM + c * stride_xC + t_in * stride_xL
        x_idx = m * stride_xM + c * stride_xC + t_in * stride_xL
        x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
        w_val = tl.load(W_ptr + c * stride_wC + k * stride_wK)
        acc += x_val * w_val

    # add bias
    bias = tl.load(BIAS_ptr + c)
    acc += bias

    # store to Out[m, c, :]
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
    # X: [M, N] row-major, W: [N, N] row-major (here N=H), Out: [M, N] row-major
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

        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xn,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wn + k_ids[:, None] * stride_wm,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )

        acc += tl.dot(x, tl.trans(w))

    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
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
        K = conv_weight.shape[1]  # 4
        # We will use float32 throughout; assume x.dtype == float32
        # 1) In-projection: y_flat = x_flat @ W^T + bias, shape (B*S, M_out)
        x_flat = x.reshape(B * S, H).contiguous()
        y_flat = torch.empty((B * S, M_out), dtype=x.dtype, device=device)

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

        # 2) Chunk y_flat into B, C, x_proj (each size H along last dim)
        # y_flat shape: (M=B*S, M_out=3*H)
        Bsz = H
        Csz = H
        xps = H  # x_proj size
        B_flat = torch.empty((B * S, Bsz), dtype=x.dtype, device=device)
        C_flat = torch.empty((B * S, Csz), dtype=x.dtype, device=device)
        X_flat = torch.empty((B * S, xps), dtype=x.dtype, device=device)

        grid_chunk = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(H, BLOCK_N))
        chunk3_kernel[grid_chunk](
            y_flat, Bsz, Csz, xps, B_flat, C_flat, X_flat,
            B * S,
            y_flat.stride(0), y_flat.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            X_flat.stride(0), X_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B_flat * X_flat, shape (B*S, H)
        Bx_flat = torch.empty((B * S, H), dtype=x.dtype, device=device)
        grid_mul = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(H, BLOCK_N))
        elementwise_mul_kernel[grid_mul](
            B_flat, X_flat, Bx_flat,
            B * S, H,
            B_flat.stride(0), B_flat.stride(1),
            X_flat.stride(0), X_flat.stride(1),
            Bx_flat.stride(0), Bx_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal conv1d: Bx -> conv_out, groups=H, K=4, PAD=3
        # Pre-pad Bx along sequence axis: left pad by PAD (3), right pad by 0
        # Bx has shape (B*S, H) row-major, conv along H? Actually conv along sequence dimension S.
        # To implement grouped conv, for each (b, h), conv along S. We can pad S dimension.
        Bx_padded = torch.empty((B * S, H), dtype=x.dtype, device=device)
        pad_left = PAD  # 3
        for m in range(B * S):
            seq = Bx_flat[m]  # (H,)
            # causal pad on left: shift by pad_left
            Bx_padded[m] = torch.nn.functional.pad(seq.unsqueeze(0), (pad_left, 0))[0]  # (H,)

        conv_out_flat = torch.empty((B * S, H), dtype=x.dtype, device=device)
        # Launch per (m, c)
        grid_conv = (B * S, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out_flat,
            B * S, H, S, PAD, K,  # Here L=S; conv along sequence
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out_flat.stride(0), conv_out_flat.stride(1), conv_out_flat.stride(2),
            BLOCK_C=1, BLOCK_T=256,
            num_warps=4, num_stages=2
        )

        # Reshape conv_out_flat to (B, S, H)
        conv_out = conv_out_flat.view(B, S, H).contiguous()

        # 5) Output gating: y = C_flat * conv_out
        # C_flat: (B*S, H), conv_out: (B, S, H)
        C_flat_expanded = C_flat.unsqueeze(1)  # (B*S, 1, H)
        conv_out_flat = conv_out.reshape(B * S, H)  # (B*S, H)
        y_gate_flat = C_flat_expanded * conv_out_flat  # (B*S, H)

        # 6) Final projection: y_gate_flat @ W_out^T + bias_out
        out_flat = torch.empty((B * S, H), dtype=x.dtype, device=device)
        BLOCK_M_out, BLOCK_N_out, BLOCK_K_out = 64, 64, 32
        grid_out = (triton.cdiv(B * S, BLOCK_M_out), triton.cdiv(H, BLOCK_N_out))
        out_proj_linear_kernel[grid_out](
            y_gate_flat, out_proj_weight, out_proj_bias, out_flat,
            B * S, H,
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
