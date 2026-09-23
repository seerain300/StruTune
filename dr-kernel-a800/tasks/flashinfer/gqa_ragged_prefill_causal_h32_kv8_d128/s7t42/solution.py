import torch
import math
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L = Q @ K^T where:
      - Q: [Nq, Hq, D]
      - K: [Nk, Hq, D] (we use Hq=32 for expanded K)
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
        acc += tl.dot(Q_tile, K_tile)  # [BLOCK_M, BLOCK_N]

    # Scale by sm_scale
    acc = acc * sm_scale

    # Store to L
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    delta: tl.int32,  # Nk - Nq
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply forward-looking causal mask: for each query q_idx, allow kv position j if j < (q_idx + 1 + delta).
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # q_idx is the start of this tile
    q_idx = pid0

    # Build causal condition: j < (q_idx + 1 + delta)
    cond = k_offsets[None, :] < (q_idx + 1 + delta)  # [1, BLOCK_N] broadcasts over rows

    # Load tile from L
    tile = tl.load(
        L_ptr + q_offsets[:, None] * (32 * Nk) + pid2 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=0.0
    )

    # Apply mask: set invalid positions to -inf
    tile = tl.where(cond, tile, -float('inf'))

    # Store back
    tl.store(
        L_ptr + q_offsets[:, None] * (32 * Nk) + pid2 * Nk + k_offsets[None, :],
        tile,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def softmax_rows_kernel(
    L_ptr, A_ptr, Nq: tl.int32, Nk: tl.int32,
    BLOCK_N: tl.constexpr
):
    """
    Row-wise softmax over Nk for each (q, head): A[q, h, :] = softmax(L[q, h, :])
    Grid: (pid0 over Nq, pid1 over heads)
    """
    pid0 = tl.program_id(0)  # q index
    pid1 = tl.program_id(1)  # head index

    row_base = pid0 * (32 * Nk) + pid1 * Nk

    # Pass 1: compute row max
    m = -float('inf')
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        tile_max = tl.max(vals, axis=0)
        m = tl.maximum(m, tile_max)

    # Pass 2: compute denominator sum
    den = 0.0
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        e = tl.exp(vals - m)
        den += tl.sum(e, axis=0)

    # Pass 3: write normalized probabilities
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        probs = tl.exp(vals - m) / den
        tl.store(A_ptr + pid0 * (32 * Nk) + pid1 * Nk + k_offsets, probs, mask=mask)


@triton.jit
def attn_dot_v_kernel(
    A_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = A @ V_expanded^T where:
      - A: [Nq, 32, Nk] (attention weights)
      - V: [Nk, 32, D] (expanded)
      - Y: [Nq, 32, D] float32
    Grid: (pid0 over Nq, pid1 over heads)
    """
    pid0 = tl.program_id(0)  # q index
    pid1 = tl.program_id(1)  # head index

    out_vec = tl.zeros((D,), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk
        a_vec = tl.load(A_ptr + pid0 * (32 * Nk) + pid1 * Nk + j_offsets, mask=mask_j, other=0.0)  # [BLOCK_N]

        # Accumulate over V across D in tiles
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask_d = d_offsets < D

            V_tile = tl.load(
                V_ptr + j_offsets[:, None] * (32 * D) + pid1 * D + d_offsets[None, :],
                mask=mask_j[:, None] & mask_d[None, :],
                other=0.0
            )  # [BLOCK_N, BLOCK_D]

            # Reduce: out_vec += sum over j of (a_vec[j] * V_tile[j, :])
            contrib = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for jj in range(BLOCK_N):
                # safe guard
                j_idx = j_start + jj
                if j_idx < Nk:
                    contrib += a_vec[j_idx] * V_tile[jj, :]
            out_vec += contrib

    # Store output vector to Y
    tl.store(Y_ptr + pid0 * (32 * D) + pid1 * D + tl.arange(0, D), out_vec, mask=True)


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    delta: tl.int32,
    BLOCK_N: tl.constexpr
):
    """
    Compute LSE per (q, head) = logsumexp(masked L along Nk) / ln(2).
    Grid: (pid0 over Nq, pid1 over heads)
    """
    pid0 = tl.program_id(0)  # q index
    pid1 = tl.program_id(1)  # head index

    row_base = pid0 * (32 * Nk) + pid1 * Nk

    m = -float('inf')
    s = 0.0
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        # Apply causal mask: allow if j < (q_idx + 1 + delta)
        cond = k_offsets[None, :] < (pid0 + 1 + delta)
        vals = tl.where(cond, vals, -float('inf'))
        tile_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
        m = new_m

    lse = tl.log(s) + m  # [1]
    lse = lse / 1.4426950408889634  # 1 / ln(2)
    tl.store(LSE_ptr + pid0 * 32 + pid1, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tuned meta-parameters to avoid shared memory issues for Nq,Nk ~ 128
        self.BLOCK_M = 32
        self.BLOCK_N = 64
        self.BLOCK_D = 64
        self.num_stages = 2
        self.num_warps = 4

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        sm_scale: float32 scalar
        Returns (output: [total_q, 32, 128], lse: [total_q, 32], both float32)
        """


def run(*args):
    return ModelNew()(*args)
