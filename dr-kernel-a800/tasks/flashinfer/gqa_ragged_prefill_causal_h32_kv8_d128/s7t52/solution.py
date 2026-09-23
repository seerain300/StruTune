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
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over Hq)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)  # head index [0..Hq-1]

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load Q tile [BLOCK_M, BLOCK_D]
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0
        )

        # Load K tile [BLOCK_N, BLOCK_D]
        K_tile = tl.load(
            K_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        acc += tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]

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
    Hq: tl.constexpr
):
    """
    In-place causal mask on L: for each (q, k, h), if k >= (q + 1 + delta) set to -inf.
    Grid: (Nq, Nk, Hq)
    delta = Nk - Nq.
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)  # head index

    q_idx = pid0
    k_idx = pid1
    delta = Nk - Nq  # valid since segment length matches

    cond = k_idx < (q_idx + 1 + delta)
    # If not cond, set to -inf
    cur = tl.load(L_ptr + q_idx * (Hq * Nk) + pid2 * Nk + k_idx)
    cur = tl.where(cond, cur, -float('inf'))
    tl.store(L_ptr + q_idx * (Hq * Nk) + pid2 * Nk + k_idx, cur)


@triton.jit
def softmax_dimN_kernel(
    Attn_ptr, Softmax_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, BLOCK_M: tl.constexpr
):
    """
    Stable softmax over Nk for each (q, head):
      - Attn: [Nq, Hq, Nk]
      - Softmax: same shape
    Grid: (pid0 over Nq tiles, pid1 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    head = pid1
    mask_q = q_offsets < Nq

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_M):
        k_offsets = j_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_k = k_offsets < Nk

        attn = tl.load(
            Attn_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        attn_max = tl.max(attn, axis=1)  # [BLOCK_M]
        new_m = tl.maximum(m, attn_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(attn - new_m[:, None]), axis=1)
        m = new_m

    inv_s = 1.0 / s
    for j_start in range(0, Nk, BLOCK_M):
        k_offsets = j_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_k = k_offsets < Nk
        attn = tl.load(
            Attn_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        out = tl.exp(attn - m[:, None]) * inv_s[:, None]
        tl.store(
            Softmax_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            out,
            mask=mask_q[:, None] & mask_k[None, :]
        )


@triton.jit
def attn_dot_v_kernel(
    Attn_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr, BLOCK_M: tl.constexpr
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

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    head = pid1
    mask_q = q_offsets < Nq

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_M):
        k_offsets = j_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_k = k_offsets < Nk

        attn_tile = tl.load(
            Attn_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )  # [BLOCK_M, BLOCK_M]

        V_tile = tl.load(
            V_ptr + k_offsets[:, None] * (Hq * D) + head * D + tl.arange(0, D),
            mask=mask_k[:, None],
            other=0.0
        )  # [BLOCK_M, D]

        acc += tl.dot(attn_tile, V_tile)  # [BLOCK_M, D]

    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + head * D + tl.arange(0, D),
        acc,
        mask=mask_q[:, None]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    sm_scale: tl.float32, BLOCK_M: tl.constexpr
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

    for j_start in range(0, Nk, BLOCK_M):
        k_offsets = j_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_k = k_offsets < Nk

        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        tile = tile * sm_scale
        tile_max = tl.max(tile, axis=1)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)
        m = new_m

    lse = tl.log(s) + m
    lse = lse / 1.4426950408889634  # 1 / ln(2)
    tl.store(LSE_ptr + q_offsets * Hq + head, lse, mask=mask_q)


class ModelNew(torch.nn.Module):
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
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        device = q.device

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        Hq = 32
        D = 128
        g = Hq // 8  # 4

        output = torch.empty((total_q, Hq, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)

        sm_scale = float(sm_scale)

        # Process each segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract segments
            q_batch = q[q_start:q_end].contiguous()  # [Nq, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous()  # [Nk, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous()  # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]

            # Expand K and V by GQA ratio
            k_expanded = k_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L [Nq, 32, Nk], float32
            L = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)

            # 1) Compute Q @ K^T scaled by sm_scale
            BLOCK_M = 32
            BLOCK_N = 64
            BLOCK_D = 64
            grid_qk = (triton.cdiv(Nq, BLOCK_M), triton.cdiv(Nk, BLOCK_N), Hq)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L,
                Nq, Nk,
                Hq=Hq, D=D,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
            )

            # 2) Apply causal mask in-place on L


def run(*args):
    return ModelNew()(*args)
