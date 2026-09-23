import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L = Q @ K^T where:
      - Q: [Nq, Hq, D]
      - K: [Nk, Hq, D] (we use expanded K with Hq=32)
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

        # Accumulate outer product
        acc += tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]

    # Scale by sm_scale
    acc = acc * sm_scale

    # Store to L: L_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :]
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply forward-look causal mask: for each (q, j), allow if j < (q + 1 + (Nk - Nq)),
    else set to -inf. Operates in-place on L_ptr of shape [Nq, 32, Nk].
    Grid: (over Nq tiles, over Nk tiles, over heads).
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Load tile
    L_tile = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=0.0
    )

    # Compute cond: allow if k_offsets < (q_offsets + 1 + delta), where delta = Nk - Nq
    delta = Nk - Nq
    q_vec = q_offsets[:, None]  # [BLOCK_M, 1]
    cond = k_offsets[None, :] < (q_vec + 1 + delta)  # [BLOCK_M, BLOCK_N]
    L_tile = tl.where(cond, L_tile, -float('inf'))

    # Store back
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        L_tile,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def softmax_rowwise_kernel(
    L_ptr, Softmax_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute softmax along Nk dimension per (q, head): Softmax[q, h, :] = softmax(L[q, h, :]).
    Store result to Softmax_ptr of shape [Nq, Hq, Nk].
    Grid: (over Nq tiles, over heads, over Nk tiles).
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_start = pid0 * BLOCK_M
    n_start = pid2 * BLOCK_N

    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Load row L: [BLOCK_M, BLOCK_N]
    L_row = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=-float('inf')
    )

    # Numerically stable softmax: subtract max per row
    row_max = tl.max(L_row, axis=1)  # [BLOCK_M]
    L_row = L_row - row_max[:, None]

    # exp and sum
    exp_row = tl.exp(L_row)
    row_sum = tl.sum(exp_row, axis=1)  # [BLOCK_M]
    softmax_row = exp_row / row_sum[:, None]  # [BLOCK_M, BLOCK_N]

    # Store to Softmax
    tl.store(
        Softmax_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
        softmax_row,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def attn_dot_v_kernel(
    Softmax_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = Softmax @ V where:
      - Softmax: [Nq, Hq, Nk]
      - V: [Nk, Hq, D]
      - Y: [Nq, Hq, D]
    Grid: (over Nq tiles, over D tiles, over heads).
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
        n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = n_offsets < Nk

        # Load Softmax tile: [BLOCK_M, BLOCK_N]
        Softmax_tile = tl.load(
            Softmax_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + n_offsets[None, :],
            mask=mask_q[:, None] & mask_n[None, :],
            other=0.0
        )

        # Load V tile: [BLOCK_N, BLOCK_D]
        V_tile = tl.load(
            V_ptr + n_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0
        )

        # Accumulate: acc += Softmax_tile @ V_tile
        acc += tl.dot(Softmax_tile, V_tile)  # [BLOCK_M, BLOCK_D]

    # Store result Y
    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable tile sizes; set to 64 for 128-dim heads
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_D = 64

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
        g = Hq // 8  # GQA ratio

        # Expand K and V along head dimension
        k_expanded = k.repeat_interleave(g, dim=1)  # [total_kv, 32, 128]
        v_expanded = v.repeat_interleave(g, dim=1)  # [total_kv, 32, 128]

        # Output and LSE tensors (float32 compute)
        output = torch.empty((total_q, Hq, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)

        # Process each segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract segments and convert to float32 for compute
            q_batch = q[q_start:q_end].to(torch.float32).contiguous()  # [Nq, 32, 128]
            k_batch = k_expanded[kv_start:kv_end].contiguous()       # [Nk, 32, 128]
            v_batch = v_expanded[kv_start:kv_end].contiguous


def run(*args):
    return ModelNew()(*args)
