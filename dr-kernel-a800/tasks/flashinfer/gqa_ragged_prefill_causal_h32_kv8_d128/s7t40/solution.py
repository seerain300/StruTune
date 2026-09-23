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
      - K: [Nk, Hq, D] (K is length Nk but used as expanded heads)
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

        # Accumulate: (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(Q_tile, tl.trans(K_tile))

    # Scale by sm_scale
    acc = acc * sm_scale

    # Store to L [Nq, Hq, Nk] row-wise
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr, Nq: tl.int32, Nk: tl.int32,
    delta: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply causal mask on L: L[q, h, k] = -inf if k >= (q + 1 + delta), else L.
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Build causal condition vector for this program
    q_idx = pid0  # since grid runs contiguous over Nq
    q_idx_vec = tl.full((BLOCK_M,), q_idx, dtype=tl.int32)
    cond = k_offsets[None, :] < (q_idx_vec[:, None] + 1 + delta)  # [BLOCK_M, BLOCK_N]

    # Load tile from L
    tile = tl.load(
        L_ptr + q_offsets[:, None] * (32 * Nk) + pid2 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=0.0
    )

    # Apply mask
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

    # We will process Nk in tiles of BLOCK_N
    row_base = pid0 * (32 * Nk) + pid1 * Nk

    # Pass 1: compute row max
    m = -float('inf')
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        # Find max in this tile
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

    # Pass 3: write normalized softmax
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        e = tl.exp(vals - m)
        out = e / den
        tl.store(A_ptr + pid0 * (32 * Nk) + pid1 * Nk + k_offsets, out, mask=mask)


@triton.jit
def attn_dot_v_kernel(
    A_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = A @ V_expanded^T where:
      - A: [Nq, Hq, Nk] float32
      - V: [Nk, Hq, D] (expanded from original 8->32)
      - Y: [Nq, Hq, D] float32
    Grid: (pid0 over Nq tiles, pid1 over D tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    d_offsets = pid1 * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    mask_q = q_offsets < Nq
    mask_d = d_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, Nk, BLOCK_N):
        k_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = k_offsets < Nk

        # Load A tile: [BLOCK_M, BLOCK_N]
        A_tile = tl.load(
            A_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_n[None, :],
            other=0.0
        )

        # Load V tile: [BLOCK_N, BLOCK_D]
        V_tile = tl.load(
            V_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0
        )

        # Accumulate
        acc += tl.dot(A_tile, V_tile)

    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :]
    )


@triton.jit
def lse_segment_row_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    BLOCK_N: tl.constexpr
):
    """
    Compute LSE per row (q, head) = logsumexp(L[q, head, :]) / ln(2).
    Grid: (pid0 over Nq, pid1 over heads)
    """
    pid0 = tl.program_id(0)  # q index
    pid1 = tl.program_id(1)  # head index

    row_base = pid0 * (32 * Nk) + pid1 * Nk

    m = -float('inf')
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        tile_max = tl.max(vals, axis=0)
        m = tl.maximum(m, tile_max)

    s = 0.0
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = k_offsets < Nk
        vals = tl.load(L_ptr + row_base + k_offsets, mask=mask, other=-float('inf'))
        e = tl.exp(vals - m)
        s += tl.sum(e, axis=0)

    lse = tl.log(s) + m  # scalar per row
    # Store to LSE_ptr [Nq, 32]
    tl.store(LSE_ptr + pid0 * 32 + pid1, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable tile sizes; chosen to balance occupancy and performance for typical dims
        self.BLOCK_M = 64   # tile over Nq
        self.BLOCK_N = 128  # tile over Nk
        self.BLOCK_D = 128  # tile over D
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

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
        Hq = self.num_qo_heads
        D = 128
        g = self.gqa_ratio

        output = torch.empty((total_q, Hq, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)

        # Convert to float32 for compute (matches original behavior)
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_batch = q_f32[q_start:q_end]                 # [Nq, 32, 128]
            k_batch = k_f32[kv_start:kv_end]              # [Nk, 8, 128]
            v_batch = v_f32[kv_start:kv_end]              # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]
            delta = Nk - Nq  # scalar int

            # GQA expansion
            k_expanded = k_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L [Nq, 32, Nk] float32
            L = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)

            # 1) Matmul: Q @ K_expanded^T
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L,
                Nq, Nk,
                Hq=Hq, D=D,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # 2) Apply causal mask
            grid_mask = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            apply_causal_mask_kernel[grid_mask](
                L, Nq, Nk, delta,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # 3) Row-wise softmax over Nk for each (q, head)
            grid_softmax = (Nq, Hq)
            A = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)
            softmax_rows_kernel[grid_softmax](
                L, A, Nq, Nk,
                BLOCK_N=self.BLOCK_N
            )

            # 4) Output = A @ V_expanded^T
            Y = torch.empty((Nq, Hq, D), dtype=torch.float32, device=device)
            grid_attn = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(D, self.BLOCK_D), Hq)
            attn_dot_v_kernel[grid_attn](
                A, v_expanded, Y,
                Nq, Nk,
                Hq=Hq, D=D,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # 5) LSE per (q, head) = logsumexp(A) / ln(2)
            grid_lse = (Nq, Hq)
            lse_segment_row_kernel[grid_lse](
                A, lse[q_start:q_start + Nq], Nq, Nk,
                BLOCK_N=self.BLOCK_N
            )

            # Store output and lse for this segment
            output[q_start:q_start + Nq] = Y
            lse[q_start:q_start + Nq] = lse[q_start:q_start + Nq]

        return output, lse


def run(*args):
    return ModelNew()(*args)
