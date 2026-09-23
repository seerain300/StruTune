import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_and_out_per_triplet_qsub(
    q_sub_ptr,         # *float32, [total_q, num_qo_heads, head_dim], contiguous
    k_ptr, v_ptr,      # *float32, [num_pages, num_kv_heads, head_dim, 1] (4D logical; we index as 3D linearly)
    qo_indptr_ptr,     # *int32, [len_indptr]
    kv_indptr_ptr,     # *int32, [len_indptr]
    kv_indices_ptr,    # *int32, [num_kv_indices]
    out_ptr,           # *float32, [total_q, num_qo_heads, head_dim], contiguous
    lse_ptr,           # *float32, [total_q, num_qo_heads], contiguous
    sm_scale,          # float32 scalar
    GQA_RATIO: tl.constexpr,   # e.g., 4
    HEAD_DIM: tl.constexpr,    # e.g., 128
    NUM_QO_HEADS: tl.constexpr,  # e.g., 32
    NUM_KV_HEADS: tl.constexpr,  # e.g., 8
    MAX_KV: tl.constexpr,        # e.g., 256
):
    # Grid is (len_indptr, total_q, num_qo_heads)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load segment starts for q and kv
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens_in_b = qo_end - qo_start
    num_kv_indices_in_b = kv_end - kv_start
    delta = num_kv_indices_in_b - num_q_tokens_in_b

    # Global query index for this triplet
    global_q_idx = qo_start + q_idx

    # Causal + length masking
    candidate_max = q_idx + 1 + delta  # scalar int
    kv_head = h // GQA_RATIO

    # Load q_sub vector: q_sub[global_q_idx, h, :]
    q_off = global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_sub = tl.load(q_sub_ptr + q_off)  # 1D vector of length HEAD_DIM

    # Initialize LSE and output vector
    lse = -float("inf")
    out_vec = [0.0] * HEAD_DIM

    # Iterate over potential KV indices with masks
    for i in tl.static_range(0, MAX_KV):
        # valid_i is a Triton boolean scalar
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)
        # Load index if valid
        idx = tl.load(kv_indices_ptr + kv_start + i, mask=valid_i, other=0)

        # Compute linear offset for k_ptr/v_ptr assuming logical 4D [num_pages, num_kv_heads, head_dim, 1]
        # We index linearly: off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

        # Load k_sub and v_sub vectors for this idx and kv_head
        k_sub = tl.load(k_ptr + off, mask=valid_i, other=0.0)  # [HEAD_DIM]
        v_sub = tl.load(v_ptr + off, mask=valid_i, other=0.0)  # [HEAD_DIM]

        # Compute dot product between q_sub and k_sub
        dot = tl.sum(q_sub * k_sub, axis=0)  # scalar

        # Scaled logits
        logits_scaled = dot * sm_scale

        # Update LSE in log space: new_lse = max(lse, logits_scaled)
        new_lse = tl.maximum(lse, logits_scaled)

        # Compute numerator and denominator for softmax
        exp_lse = tl.exp(lse)
        exp_new = tl.exp(new_lse)
        # For positions with valid_i True, numerator = exp(logits_scaled - new_lse), else 0
        numerator = tl.exp(logits_scaled - new_lse)  # scalar
        # Mask numerator by valid_i
        numerator = numerator * valid_i.to(tl.float32)  # multiply by 0/1 scalar

        # denominator = exp(lse) + numerator * exp(-logits_scaled)
        # But since numerator is 0 when invalid, we can compute directly with mask. Simpler: compute valid count
        # However, Triton doesn't have per-element reduction here. To keep it simple and correct, we recompute lse after update.

        # Re-compute logsumexp over two points: current (lse, 0) and new (logits_scaled, numerator)
        # Use stable update:
        # If new_lse > lse: out_vec += numerator * v_sub; lse = new_lse
        # Else: out_vec unchanged; lse stays
        # Implement branch: compute contrib
        contrib = numerator * v_sub  # vector
        out_vec += tl.where(new_lse > lse, contrib, contrib * 0.0)
        lse = new_lse

    # Store output for this (b, q_idx, h)
    out_off = global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])

    # Store LSE for this (b, q_idx, h)
    tl.store(lse_ptr + global_q_idx * NUM_QO_HEADS + h, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.GQA_RATIO = self.num_qo_heads // self.num_kv_heads  # 4
        self.MAX_KV = 256  # covers typical candidate_max safely

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are CUDA and contiguous, proper dtypes
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        q_f32 = q.to(torch.float32).contiguous()
        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        assert num_qo_heads == self.num_qo_heads and head_dim == self.head_dim, \
            f"q shape must be [total_q, {self.num_qo_heads}, {self.head_dim}]"

        # Precompute q_sub buffer: [total_q, num_qo_heads, head_dim]
        q_sub = q_f32  # contiguous

        # Squeeze the middle dim (batch) from k_cache and v_cache
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, num_kv_heads, head_dim]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        len_indptr = qo_indptr.shape[0]
        # Allocate outputs (float32 for computation; cast later to bfloat16)
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Ensure int32 indptr and indices
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int32)

        # Launch Triton kernel: one program per (b, q_idx, h)
        grid = (len_indptr, total_q, num_qo_heads)
        _compute_lse_and_out_per_triplet_qsub[grid](
            q_sub, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices, out_f32, lse_f32, sm_scale,
            GQA_RATIO=self.GQA_RATIO, HEAD_DIM=self.head_dim, NUM_QO_HEADS=self.num_qo_heads, NUM_KV_HEADS=self.num_kv_heads, MAX_KV=self.MAX_KV,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 for final result; lse remains float32
        output_bf16 = out_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
