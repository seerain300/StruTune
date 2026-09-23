import torch
import math
import triton
import triton.language as tl


@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.int32, D: tl.int32,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L = Q @ K^T where:
      - Q: [Nq, Hq, D]
      - K: [Nk, Hq, D]
      - L: [Nq, Hq, Nk] float32
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles)
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid1 = tl.program_id(1)  # tile over Nk

    q_start = pid0 * BLOCK_M
    k_start = pid1 * BLOCK_N

    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Accumulator [BLOCK_M, BLOCK_N], compute for all heads Hq
    # We use a Python loop over heads (Hq is constexpr-like here)
    for h in range(0, 32):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Loop over D dimension
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask_d = d_offsets < D

            # Load Q tile [BLOCK_M, BLOCK_D] for head h
            Q_tile = tl.load(
                Q_ptr + q_offsets[:, None] * (Hq * D) + h * D + d_offsets[None, :],
                mask=mask_q[:, None] & mask_d[None, :],
                other=0.0
            )

            # Load K tile [BLOCK_D, BLOCK_N] for head h, then transpose to [BLOCK_N, BLOCK_D]
            K_tile = tl.load(
                K_ptr + k_offsets[None, :] * (Hq * D) + h * D + d_offsets[:, None],
                mask=mask_d[:, None] & mask_k[None, :],
                other=0.0
            )
            K_tile_T = tl.trans(K_tile)  # [BLOCK_N, BLOCK_D]

            # Outer product accumulation: [BLOCK_M, BLOCK_D] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_N]
            acc += tl.dot(Q_tile, K_tile_T)

        # Scale logits
        acc = acc * sm_scale

        # Store L: L[q, h, k] -> offset = q*(Hq*Nk) + h*Nk + k
        tl.store(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + k_offsets[None, :],
            acc,
            mask=mask_q[:, None] & mask_k[None, :]
        )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.int32,
    delta: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply causal mask to L: for each (q, k, h), if k >= (q + 1 + delta) set L to -inf.
    Grid: (tiles over Nq, tiles over Nk, heads)
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid1 = tl.program_id(1)  # tile over Nk
    pid2 = tl.program_id(2)  # head

    q_start = pid0 * BLOCK_M
    k_start = pid1 * BLOCK_N

    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Load current L tile
    L_tile = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=0.0
    )

    # Build mask: allow if k < (q + 1 + delta)
    q_vec = q_offsets[:, None]  # [BLOCK_M, 1]
    cond = k_offsets[None, :] < (q_vec + 1 + delta)  # [BLOCK_M, BLOCK_N]

    # Set invalid to -inf
    L_tile = tl.where(cond, L_tile, -float('inf'))

    # Store back
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        L_tile,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def softmax_rowwise_kernel(
    L_ptr, Soft_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Row-wise softmax over Nk for each q and head h:
      - L_ptr: input logits [Nq, Hq, Nk]
      - Soft_ptr: output softmax [Nq, Hq, Nk]
    Grid: (tiles over Nq, heads)
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid2 = tl.program_id(1)  # head

    q_start = pid0 * BLOCK_M

    for q in range(q_start, q_start + BLOCK_M):
        if q >= Nq:
            break
        # Compute max across Nk
        m = -float('inf')
        for k_start in range(0, Nk, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            mask_k = k_offsets < Nk
            L_vec = tl.load(
                L_ptr + q * (Hq * Nk) + pid2 * Nk + k_offsets,
                mask=mask_k,
                other=-float('inf')
            )
            tile_max = tl.max(L_vec, axis=0)
            m = tl.maximum(m, tile_max)

        # Compute sum of exp(L - m)
        s = 0.0
        for k_start in range(0, Nk, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            mask_k = k_offsets < Nk
            L_vec = tl.load(
                L_ptr + q * (Hq * Nk) + pid2 * Nk + k_offsets,
                mask=mask_k,
                other=-float('inf')
            )
            s += tl.sum(tl.exp(L_vec - m), axis=0)

        # Write softmax
        for k_start in range(0, Nk, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            mask_k = k_offsets < Nk
            L_vec = tl.load(
                L_ptr + q * (Hq * Nk) + pid2 * Nk + k_offsets,
                mask=mask_k,
                other=-float('inf')
            )
            soft_vec = tl.exp(L_vec - m) / s
            tl.store(
                Soft_ptr + q * (Hq * Nk) + pid2 * Nk + k_offsets,
                soft_vec,
                mask=mask_k
            )


@triton.jit
def attn_dot_v_kernel(
    Soft_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.int32, Hq: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = Soft @ V, where:
      - Soft: [Nq, Hq, Nk]
      - V: [Nk, Hq, D] (expanded V with 32 heads)
      - Y: [Nq, Hq, D]
    Grid: (tiles over Nq, heads, tiles over D)
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid1 = tl.program_id(1)  # head
    pid2 = tl.program_id(2)  # tile over D

    q_start = pid0 * BLOCK_M
    d_start = pid2 * BLOCK_D

    q_offsets = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    mask_q = q_offsets < Nq
    mask_d = d_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Loop over Nk in chunks
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk

        # Load Soft tile [BLOCK_M, BLOCK_N]
        Soft_tile = tl.load(
            Soft_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )

        # Load V tile [BLOCK_N, BLOCK_D]
        V_tile = tl.load(
            V_ptr + k_offsets[:, None] * (Hq * D) + pid1 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        # Accumulate: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        acc += tl.dot(Soft_tile, V_tile)

    # Store Y
    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid1 * D + d_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute LSE = logsumexp(L, dim=Nk) for each q and head h, then divide by ln(2).
    LSE_ptr: [Nq, Hq] float32. Grid: (tiles over Nq, heads).
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid1 = tl.program_id(1)  # head

    q_start = pid0 * BLOCK_M

    for q in range(q_start, q_start + BLOCK_M):
        if q >= Nq:
            break
        m = -float('inf')
        s = 0.0
        for k_start in range(0, Nk, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            mask_k = k_offsets < Nk
            L_vec = tl.load(
                L_ptr + q * (Hq * Nk) + pid1 * Nk + k_offsets,
                mask=mask_k,
                other=-float('inf')
            )
            tile_max = tl.max(L_vec, axis=0)
            new_m = tl.maximum(m, tile_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(L_vec - new_m), axis=0)
            m = new_m
        lse_val = tl.log(s) + m  # logsumexp
        lse_val = lse_val / 1.4426950408889634  # ln(2)
        tl.store(LSE_ptr + q * Hq + pid1, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes tuned for D=128
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_D = 128

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
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        device = q.device

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        Hq = 32
        D = 128
        g = Hq // 8  # GQA ratio

        # Output and LSE tensors (float32 compute)
        output = torch.empty((total_q, Hq, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)

        # Precompute expanded K and V
        k_expanded = k.repeat_interleave(g, dim=1)  # [total_kv, 32, 128]
        v_expanded = v.repeat_interleave(g, dim=1)  # [total_kv, 32, 128]

        # Iterate segments
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
            v_batch = v_expanded[kv_start:kv_end].contiguous()       # [Nk, 32, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]

            # Allocate logits buffer L_tmp [Nq, 32, Nk] float32
            L_tmp = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)

            # 1) Matmul: L_tmp = q_batch @ k_batch^T (k_batch has 32 heads already)
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N))
            matmul_qk_kernel[grid_qk](
                q_batch, k_batch, L_tmp,
                Nq, Nk, Hq, D,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # 2) Apply causal mask: delta = Nk - Nq
            delta = Nk - Nq
            grid_mask = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            apply_causal_mask_kernel[grid_mask](
                L_tmp,
                Nq, Nk, Hq,
                delta=delta,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # 3) Softmax along Nk, per (q, head)
            Soft_tmp = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)
            grid_softmax = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            softmax_rowwise_kernel[grid_softmax](
                L_tmp, Soft_tmp,
                Nq, Nk, Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # 4) Output Y = Soft_tmp @ v_batch
            Y_tmp = torch.empty((Nq, Hq, D), dtype=torch.float32, device=device)
            grid_attn = (triton.cdiv(Nq, self.BLOCK_M), Hq, triton.cdiv(D, self.BLOCK_D))
            attn_dot_v_kernel[grid_attn](
                Soft_tmp, v_batch, Y_tmp,
                Nq, Nk, D, Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_D=self.BLOCK_D
            )
            output[q_start:q_end] = Y_tmp

            # 5) LSE per (q, head): logsumexp(L_tmp, dim=Nk) / ln(2)
            grid_lse = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            lse_segment_kernel[grid_lse](
                L_tmp, lse[q_start:q_end],
                Nq, Nk, Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
