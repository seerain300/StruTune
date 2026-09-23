import torch
import math
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L = Q @ K^T where:
      - Q: [Nq, Hq, D]
      - K: [Nk, Hq, D]
      - L: [Nq, Hq, Nk] float32
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load Q tile: [BLOCK_M, BLOCK_D]
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0
        )

        # Load K tile: [BLOCK_N, BLOCK_D]
        K_tile = tl.load(
            K_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        # Outer product accumulation
        acc += tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]

    # Store to L: [Nq, Hq, Nk]
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    delta: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply forward-looking causal mask to logits L: allow kv if kv < (q + 1 + delta),
    else set to -inf. Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads).
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Load current logits tile [BLOCK_M, BLOCK_N]
    tile = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=-float('inf')
    )

    # Compute causal condition: kv < (q + 1 + delta)
    q_idx = pid0 * BLOCK_M  # scalar index of first q in this tile
    cond = k_offsets[None, :] < (q_idx + 1 + delta)  # broadcast q_idx across k tile

    # Set non-causal to -inf
    tile = tl.where(cond, tile, -float('inf'))

    # Store back
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        tile,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def softmax_dimN_kernel(
    In_ptr, Out_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Stable softmax along Nk (dim=2) for each q and head:
      - In_ptr: [Nq, Hq, Nk]
      - Out_ptr: same shape
    Grid: (pid0 over Nq tiles, pid1 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    head = pid1
    mask_q = q_offsets < Nq

    # Pass 1: compute max per q, head
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    for j_start in range(0, Nk, BLOCK_N):
        k_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            In_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        tile_max = tl.max(tile, axis=1)  # [BLOCK_M]
        m = tl.maximum(m, tile_max)

    # Pass 2: compute sum of exp(tile - m)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for j_start in range(0, Nk, BLOCK_N):
        k_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            In_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        s += tl.sum(tl.exp(tile - m[:, None]), axis=1)

    inv_s = 1.0 / s

    # Pass 3: write normalized outputs
    for j_start in range(0, Nk, BLOCK_N):
        k_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            In_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        out = tl.exp(tile - m[:, None]) * inv_s[:, None]
        tl.store(
            Out_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            out,
            mask=mask_q[:, None] & mask_k[None, :]
        )


@triton.jit
def attn_dot_v_kernel(
    Attn_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = Attn @ V where:
      - Attn: [Nq, Hq, Nk]
      - V: [Nk, Hq, D]
      - Y: [Nq, Hq, D]
    Grid: (pid0 over Nq tiles, pid1 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    head = pid1
    mask_q = q_offsets < Nq

    acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk

        # Load Attn tile [BLOCK_N, 1]
        attn_tile = tl.load(
            Attn_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + j_offsets[None, :],
            mask=mask_q[:, None] & mask_j[None, :],
            other=0.0
        )

        # Load V tile [BLOCK_N, D]
        V_tile = tl.load(
            V_ptr + j_offsets[:, None] * (Hq * D) + head * D + tl.arange(0, D),
            mask=mask_j[:, None],
            other=0.0
        )

        # Accumulate
        acc += attn_tile * V_tile  # [BLOCK_N, D]

    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + head * D + tl.arange(0, D),
        acc,
        mask=mask_q[:, None]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    sm_scale: tl.float32, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute LSE per (q, head) = logsumexp(L * sm_scale) along Nk, divided by ln(2).
    Grid: (pid0 over Nq tiles, pid1 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    head = pid1
    mask_q = q_offsets < Nq

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_N):
        k_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk

        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        # Scale
        tile = tile * sm_scale
        # Row-wise max
        tile_max = tl.max(tile, axis=1)  # [BLOCK_M]
        new_m = tl.maximum(m, tile_max)
        # Update sum: s = s * exp(m - new_m) + sum(exp(tile - new_m))
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)
        m = new_m

    lse = tl.log(s) + m  # logsumexp
    # Divide by ln(2)
    lse = lse / 1.4426950408889634  # 1 / ln(2)
    tl.store(LSE_ptr + q_offsets * Hq + head, lse, mask=mask_q)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable tiles


def run(*args):
    return ModelNew()(*args)
