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
      - K: [Nk, Hq, D] (K is length Nk but we use expanded heads)
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

    # Scale by sm_scale (as in original)
    acc = acc * sm_scale

    # Store to L: L[q, h, k]
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def softmax_Nk_kernel(
    L_ptr, S_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute softmax along Nk for each (q, head):
      - L: [Nq, Hq, Nk] float32
      - S: [Nq, Hq] float32 (we store as linear index q*Hq + head)
    Grid: (pid0 over Nq tiles, pid1 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq
    head = pid1

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )
        tile_max = tl.max(tile, axis=1)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)
        m = new_m

    soft = tl.exp(s)  # [BLOCK_M]
    # Store to S as linear [Nq, Hq]
    for i in range(BLOCK_M):
        if q_offsets[i] < Nq:
            idx = q_offsets[i] * Hq + pid1
            tl.store(S_ptr + idx, soft[i])


@triton.jit
def attn_dot_v_kernel(
    S_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = S @ V_expanded:
      - S: [Nq, Hq] float32 (softmax along Nk per (q, head))
      - V_expanded: [Nk, Hq, D] float32
      - Y: [Nq, Hq, D] float32
    Grid: (pid0 tiles over Nq, pid1 head id)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Accumulate over k
    for k_idx in range(0, Nk):
        # Load s[q, head] as a vector for the tile
        s_vals = tl.load(S_ptr + q_offsets * Hq + pid1, mask=mask_q, other=0.0)  # [BLOCK_M]
        # Load V_expanded[k, head, :]
        V_row = tl.load(V_ptr + k_idx * (Hq * D) + pid1 * D + tl.arange(0, D), mask=True, other=0.0)  # [D]
        acc += s_vals[:, None] * V_row[None, :]

    # Store Y[q, head, :]
    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid1 * D + tl.arange(0, D)[None, :],
        acc,
        mask=mask_q[:, None]
    )


@triton.jit
def lse_reduce_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute LSE per (q, head) = logsumexp(L) along Nk, divided by ln(2).
    Write to LSE_ptr [Nq, Hq] float32 (forward will slice q_start:q_end).
    Grid: (pid0 tiles over Nq, pid1 head id)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq
    head = pid1

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + head * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )
        tile_max = tl.max(tile, axis=1)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)
        m = new_m

    lse = tl.log(s) + m  # [BLOCK_M]
    # Divide by ln(2)
    log2 = 0.6931471805599453
    lse = lse / log2

    for i in range(BLOCK_M):
        if q_offsets[i] < Nq:
            idx = q_offsets[i] * Hq + pid1
            tl.store(LSE_ptr + idx, lse[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable block sizes (can be adjusted based on workload)
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_D = 64
        self.lse_BLOCK_M = 128
        self.lse_BLOCK_N = 64

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
        num_kv_heads = 8

        output = torch.empty((total_q, Hq, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=q.device)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract segments and cast to float32 for compute
            q_batch = q[q_start:q_end].to(torch.float32)  # [Nq, 32, 128]
            k_batch = k[kv_start:kv_end].to(torch.float32)  # [Nk, 8, 128]
            v_batch = v[kv_start:kv_end].to(torch.float32)  # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]

            # GQA expansion (heads and dim)
            k_expanded = k_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L_tmp [Nq, 32, Nk] float32
            L_tmp = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=q.device)

            # Launch Triton matmul kernel: Q=q_batch, K=k_expanded, L=L_tmp
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L_tmp,
                Nq, Nk,
                Hq=Hq, D=D,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # Softmax along Nk for each (q, head)
            S = torch.empty((Nq, Hq), dtype=torch.float32, device=q.device)  # we'll store as linearized
            grid_softmax = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            softmax_Nk_kernel[grid_softmax](
                L_tmp, S,
                Nq, Nk, Hq=Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # Output Y = S @ V_expanded
            Y = torch.empty((Nq, Hq, D), dtype=torch.float32, device=q.device)
            grid_attn = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            attn_dot_v_kernel[grid_attn](
                S, v_expanded, Y,
                Nq, Nk, D=D,
                Hq=Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_D=self.BLOCK_D
            )
            # Store into output slice
            output[q_start:q_end] = Y

            # LSE for this segment per head: compute on logits L_tmp
            grid_lse = (triton.cdiv(Nq, self.lse_BLOCK_M), Hq)
            lse_reduce_kernel[grid_lse](
                L_tmp, lse[q_start:q_end],
                Nq, Nk, Hq=Hq,
                BLOCK_M=self.lse_BLOCK_M, BLOCK_N=self.lse_BLOCK_N
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
