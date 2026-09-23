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
      - K: [Nk, Hq, D] (expanded heads via repeat_interleave)
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

    # Scale by sm_scale
    acc = acc * sm_scale

    # Store to L
    L_ptrs = L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :]
    tl.store(L_ptrs, acc, mask=mask_q[:, None] & mask_k[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as per original assertions
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)
        self.inv_ln2 = 1.0 / math.log(2.0)  # 1/ln(2)

        # Triton tuning parameters
        self.BLOCK_M = 32
        self.BLOCK_N = 64
        self.BLOCK_D = 64  # D=128

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."
        device = q.device

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 1, "len_indptr must be at least 1"

        # Prepare outputs (float32 compute, cast later)
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Convert to float32 for compute (matches original behavior)
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

            # Extract segments
            q_batch = q_f32[q_start:q_end]                 # [Nq, 32, 128]
            k_batch = k_f32[kv_start:kv_end]              # [Nk, 8, 128]
            v_batch = v_f32[kv_start:kv_end]              # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]

            # GQA expansion
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L_tmp [Nq, 32, Nk] float32
            L_tmp = torch.empty((Nq, self.num_qo_heads, Nk), dtype=torch.float32, device=device)

            # Launch Triton matmul kernel: Q=q_batch, K=k_expanded, L=L_tmp
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), self.num_qo_heads)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L_tmp,
                Nq, Nk,
                Hq=self.num_qo_heads, D=self.head_dim,
                sm_scale=float(sm_scale) if sm_scale is not None else self.sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D,
            )

            # Apply causal mask: for each query q_idx, allow attention to KV positions j where j < (q_idx + 1 + delta), with delta = Nk - Nq
            # delta may be negative (e.g., Nk < Nq), that's fine; mask becomes always true
            delta = Nk - Nq

            # Build indices for mask
            q_idx = torch.arange(Nq, device=device)               # [Nq]
            kv_idx = torch.arange(Nk, device=device)             # [Nk]

            # Condition: kv < (q + 1 + delta)
            mask = (kv_idx[None, :] < (q_idx[:, None] + 1 + delta))  # [Nq, Nk]

            # Apply mask: set invalid entries to -inf
            L_masked = L_tmp.masked_fill(~mask, -float('inf'))

            # Scale by sm_scale for softmax
            logits_scaled = L_masked / (float(sm_scale) if sm_scale is not None else self.sm_scale)

            # Softmax along the KV tokens (last dim)
            attn = torch.softmax(logits_scaled, dim=-1)  # [Nq, 32, Nk], float32

            # Final output: attn @ v_expanded (einsum qhk,khd->qhd)
            # v_expanded: [Nk, 32, 128]
            output_batch = torch.einsum('qhk,khd->qhd', attn, v_expanded)  # [Nq, 32, 128], float32
            output[q_start:q_end] = output_batch

            # Compute lse (logsumexp over masked logits, divided by ln(2)) for this segment/heads
            # lse = logsumexp(L_masked) / ln(2)
            # Note: masked entries are -inf, so they don't contribute to max/sum.
            m = torch.max(L_masked)  # [scalar]
            # sum exp(L - m) over all elements
            sum_exp = torch.sum(torch.exp(L_masked - m))
            lse_batch = torch.log(1.0 + sum_exp) * self.inv_ln2  # divide by ln(2) via 1/ln(2)
            # Store lse for all queries in this segment
            lse[q_start:q_end] = lse_batch

        # Return output (cast to bfloat16 to match original) and lse
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
