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
      - Q: [Nq, Hq, D], float32 (we pass q expanded to 32 heads)
      - K: [Nk, Hq, D], float32 (expanded K from kv heads)
      - L: [Nq, Hq, Nk], float32
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)  # head index in 0..Hq-1

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load Q tile: [BLOCK_M, BLOCK_D] from Q[pid2]
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0
        )
        # Load K tile: [BLOCK_N, BLOCK_D] from K[:, pid2]
        K_tile = tl.load(
            K_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        acc += tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]

    acc = acc * sm_scale
    # Store: L[pid0, pid2, pid1] tile
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, OutLSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute LSE for a segment:
      - L: [Nq, Hq, Nk], float32, contiguous
      - OutLSE: [Nq, Hq], float32
    Tiling over Nq and Nk; per-program reduces along Nk to produce per-(q,head) LSE.
    """
    # One program per (q tile, head)
    pid0 = tl.program_id(0)  # tile id along Nq
    pid1 = tl.program_id(1)  # head id in 0..Hq-1

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    # Initialize max and sum for logsumexp
    max_vec = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    sum_vec = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk

        # Load L tile: [BLOCK_M, BLOCK_N] for head pid1
        L_tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-1e20
        )

        # Reduce along K to get per-q max and sum
        tile_max = tl.max(L_tile, axis=1)  # [BLOCK_M]
        max_vec = tl.maximum(max_vec, tile_max)

        L_shift = L_tile - max_vec[:, None]
        sum_vec += tl.sum(tl.exp(L_shift), axis=1)

    # Compute logsumexp: log(sum) + max; then divide by ln(2)
    lse_vec = tl.log(sum_vec) + max_vec  # [BLOCK_M]
    ln2 = 0.6931471805599453
    lse_vec = lse_vec / ln2

    # Store out
    tl.store(
        OutLSE_ptr + q_offsets * Hq + pid1,
        lse_vec,
        mask=mask_q
    )


class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_M=32, BLOCK_N=64, BLOCK_D=64):
        super().__init__()
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_D = BLOCK_D

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized version:
          - Q: [total_q, 32, 128] bfloat16 (assertions: 32 heads, 128 dim)
          - K: [total_kv, 8, 128] bfloat16
          - V: [total_kv, 8, 128] bfloat16
          - qo_indptr: [len_indptr], int32 (counts of tokens per batch segment for Q)
          - kv_indptr: [len_indptr], int32 (counts of tokens per batch segment for K/V)
          - sm_scale: float32 scalar (1/sqrt(128) in original)
        Returns (output: [total_q, 32, 128] bfloat16, lse: [total_q, 32] float32)
        """
        # Constants from the original assertions
        Hq = 32
        Hkv = 8
        D = 128
        assert Hq == 32 and Hkv == 8 and D == 128, "Assumed dimensions: 32 qo_heads, 8 kv_heads, 128 dim"
        g = Hq // Hkv  # 4 (GQA ratio)

        total_q = q.shape[0]
        total_kv = k.shape[0]
        len_indptr = qo_indptr.shape[0]
        device = q.device

        # Output buffers (float32 for compute, cast back bfloat16)
        output = torch.empty((total_q, Hq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)

        # Prepare float32 views for Triton compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Process each batch segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_batch = q_f32[q_start:q_end]         # [Nq, 32, 128]
            k_batch = k_f32[kv_start:kv_end]       # [Nk, 8, 128]
            v_batch = v_f32[kv_start:kv_end]       # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]

            # GQA expansion: K and V to 32 heads
            k_expanded = k_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]

            # Allocate logits [Nq, 32, Nk] float32
            L = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)

            # Launch Triton matmul kernel: Q=q_batch, K=k_expanded, L=L
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L,
                Nq, Nk,
                Hq=Hq, D=D,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # Apply causal mask using PyTorch (GPU). Condition: kv < (q_idx + 1 + delta)
            # delta = Nk - Nq
            delta = Nk - Nq
            if delta < 0:
                # Typically Nk >= Nq; but guard against unusual cases
                mask = torch.ones((Nq, Hq, Nk), dtype=torch.bool, device=device)
            else:
                q_idx = torch.arange(Nq, device=device)          # [Nq]
                kv_idx = torch.arange(Nk, device=device)        # [Nk]
                mask = (kv_idx[None, None, :] < (q_idx[:, None] + 1 + delta))  # [Nq, 1, Nk] -> broadcast to [Nq, Hq, Nk]
            L = L.masked_fill(~mask, -float('inf'))

            # Softmax over K dimension (per (q, head)) and multiply by V
            attn = torch.softmax(L, dim=-1)  # [Nq, 32, Nk]
            output_batch = torch.einsum('qhk,khd->qhd', attn, v_expanded)  # [Nq, 32, 128]

            # Store results for this segment
            output[q_start:q_start + Nq] = output_batch.to(torch.bfloat16)

            # Compute LSE via Triton reduction per segment
            # lse[q_start:q_start+Nq, :] = logsumexp(L) / ln(2)
            grid_lse = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            lse_segment_kernel[grid_lse](
                L, lse[q_start:q_start + Nq],
                Nq, Nk,
                Hq=Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
