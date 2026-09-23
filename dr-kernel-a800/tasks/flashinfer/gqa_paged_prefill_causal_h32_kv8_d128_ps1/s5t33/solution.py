import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_and_out_per_triplet(
    q_ptr,                 # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr, v_ptr,          # *float32, [num_pages, num_kv_heads, head_dim, 1] (logical 4D)
    qo_indptr_ptr,         # *int32, [len_indptr]
    kv_indptr_ptr,         # *int32, [len_indptr]
    kv_indices_ptr,        # *int32, [num_kv_indices]
    out_ptr,               # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,               # *float32, [total_q, num_qo_heads]
    sm_scale,              # float32 scalar
    NUM_QO_HEADS: tl.constexpr,  # e.g., 32
    NUM_KV_HEADS: tl.constexpr,  # e.g., 8
    HEAD_DIM: tl.constexpr,      # e.g., 128
    GQA_RATIO: tl.constexpr,     # e.g., 4
    MAX_KV: tl.constexpr,        # e.g., 128 (covers head_dim)
):
    # Grid is (len_indptr, total_q, num_qo_heads)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load segment starts
    qo_start = tl.load(qo_indptr_ptr + b)  # int32
    qo_end = tl.load(qo_indptr_ptr + b + 1)  # int32
    kv_start = tl.load(kv_indptr_ptr + b)  # int32
    kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

    # Number of tokens in this batch segment
    num_q_tokens_in_b = qo_end - qo_start  # int32
    num_kv_indices_in_b = kv_end - kv_start  # int32

    # Causal + length masking
    candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens_in_b)  # int32
    kv_head = h // GQA_RATIO  # 0..7

    # Global query index
    global_q_idx = qo_start + q_idx  # int32

    # Load q_sub vector: q[global_q_idx, h, :]
    q_off = global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_sub = tl.load(q_ptr + q_off)  # [HEAD_DIM] float32 vector

    # Initialize lse and output vector
    lse_acc = -float("inf")  # float32
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Process up to MAX_KV entries with masks
    for i in tl.static_range(0, MAX_KV):
        # valid_i conditions
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & ((kv_start + i) < kv_end)
        # If not valid, skip
        if not valid_i:
            continue

        # Load idx from kv_indices_ptr
        idx = tl.load(kv_indices_ptr + kv_start + i)  # int32

        # Compute linear offset for k_ptr/v_ptr which are logical [num_pages, num_kv_heads, head_dim, 1]
        off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

        # Load k_sub and v_sub (both [HEAD_DIM])
        k_sub = tl.load(k_ptr + off)  # [HEAD_DIM] float32
        v_sub = tl.load(v_ptr + off)  # [HEAD_DIM] float32

        # Dot product: q_sub @ k_sub (scalar)
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_sub[j] * k_sub[j]

        # Scale logits
        logits_scaled = dot * sm_scale  # float32 scalar

        # Update lse with logsumexp (stable): new_lse = max(old, logits_scaled)
        new_lse = tl.maximum(lse_acc, logits_scaled)
        numerator = tl.exp(logits_scaled - new_lse)
        denominator = tl.exp(lse_acc - new_lse) + numerator
        lse_acc = new_lse

        # Compute softmax probability for this key
        prob = numerator / denominator  # scalar float32

        # Accumulate output vector: out += prob * v_sub
        for j in tl.static_range(0, HEAD_DIM):
            out_vec[j] += prob * v_sub[j]

    # Store output for this (b, q_idx, h)
    out_off = global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * NUM_QO_HEADS + h
    tl.store(lse_ptr + lse_off, lse_acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.max_kv = 128  # safe upper bound for candidate_max (covers head_dim)

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        # Reconstruct total_q from indptrs
        total_q = int(qo_indptr[-1].item())

        # Preprocess inputs
        q_f32 = q.to(torch.float32).contiguous()  # [*, 32, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]

        # Allocate outputs
        out_f32 = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: grid over (len_indptr, total_q, num_qo_heads)
        grid = (len_indptr, total_q, self.num_qo_heads)
        _compute_lse_and_out_per_triplet[grid](
            q_f32, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices, out_f32, lse_f32, float(sm_scale),
            NUM_QO_HEADS=self.num_qo_heads,
            NUM_KV_HEADS=self.num_kv_heads,
            HEAD_DIM=self.head_dim,
            GQA_RATIO=self.gqa_ratio,
            MAX_KV=self.max_kv,
            num_warps=4, num_stages=2
        )

        # Return output as bfloat16 (to match original behavior) and lse as float32
        return out_f32.to(torch.bfloat16), lse_f32


def run(*args):
    return ModelNew()(*args)
