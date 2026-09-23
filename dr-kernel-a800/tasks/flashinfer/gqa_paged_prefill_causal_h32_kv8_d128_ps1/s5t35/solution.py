import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_one_triplet_kernel(
    q_ptr,            # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr, v_ptr,     # *float32, [num_pages, num_kv_heads, head_dim] (already squeezed(1))
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,          # *float32, [total_q, num_qo_heads]
    GQA_RATIO: tl.constexpr,   # e.g., 4
    MAX_KV: tl.constexpr,      # e.g., 256
    sm_scale: tl.float32,      # scaling factor
):
    # Program ids: each program handles one (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load qo_indptr[b] and qo_indptr[b+1] to get sequence range for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)

    # GQA mapping: query head h uses kv head kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # Compute q_sub vector (head h) for this query position
    global_q_idx = qo_start + q_idx  # guaranteed < qo_end by grid
    base_q = global_q_idx * (32 * 128) + h * 128
    q_sub = tl.load(q_ptr + base_q + tl.arange(0, 128))  # [128]

    # Determine num_kv_indices_in_b and kv range
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)
    num_q_tokens_in_b = qo_end - qo_start
    num_kv_indices_in_b = kv_end - kv_start

    # candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens_in_b)
    delta = num_kv_indices_in_b - num_q_tokens_in_b
    candidate_max = (q_idx + 1) + delta  # scalar Triton int

    # Prepare vector to hold all logits across i
    logits_vec = tl.full((MAX_KV,), -float("inf"), dtype=tl.float32)

    # First pass: fill logits_vec for valid i
    for i in tl.static_range(0, MAX_KV):
        # validity: i < candidate_max and i < num_kv_indices_in_b and kv_start + i < kv_end
        cond_i = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)

        # Compute idx; if cond_i is False, we still need idx for pointer arithmetic, but loads will be masked
        idx = tl.load(kv_indices_ptr + kv_start + i)

        # Compute offsets for k and v (squeezed shape [num_pages, num_kv_heads, head_dim])
        k_off = idx * (8 * 128) + kv_head * 128
        v_off = idx * (8 * 128) + kv_head * 128

        # Load k_vec and v_vec
        k_vec = tl.load(k_ptr + k_off + tl.arange(0, 128))
        v_vec = tl.load(v_ptr + v_off + tl.arange(0, 128))

        # Compute dot product between q_sub and k_vec
        logits_i = 0.0
        for j in tl.static_range(0, 128):
            logits_i += q_sub[j] * k_vec[j]
        logits_i = logits_i * sm_scale

        # If invalid, set logits_i to -inf so it doesn't contribute to lse
        logits_vec[i] = tl.where(cond_i, logits_i, -float("inf"))

    # Compute lse = logsumexp(logits_vec)
    m = -float("inf")
    for i in tl.static_range(0, MAX_KV):
        m = tl.maximum(m, logits_vec[i])
    s = 0.0
    for i in tl.static_range(0, MAX_KV):
        s += tl.exp(logits_vec[i] - m)
    lse_val = tl.log(s) + m

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector by softmax over logits_scaled across valid i
    out_vec = tl.zeros(128, dtype=tl.float32)
    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)

        idx = tl.load(kv_indices_ptr + kv_start + i)
        k_off = idx * (8 * 128) + kv_head * 128
        v_off = idx * (8 * 128) + kv_head * 128

        k_vec = tl.load(k_ptr + k_off + tl.arange(0, 128))
        v_vec = tl.load(v_ptr + v_off + tl.arange(0, 128))

        logits_i = 0.0
        for j in tl.static_range(0, 128):
            logits_i += q_sub[j] * k_vec[j]
        logits_scaled_i = logits_i * sm_scale

        # Compute total_exp = sum(exp(logits_scaled)) over all valid i
        total_exp = 0.0
        for t in tl.static_range(0, MAX_KV):
            cond_t = (t < candidate_max) & (t < num_kv_indices_in_b) & (kv_start + t < kv_end)
            idx_t = tl.load(kv_indices_ptr + kv_start + t)
            k_off_t = idx_t * (8 * 128) + kv_head * 128
            k_vec_t = tl.load(k_ptr + k_off_t + tl.arange(0, 128))
            v_off_t = idx_t * (8 * 128) + kv_head * 128
            v_vec_t = tl.load(v_ptr + v_off_t + tl.arange(0, 128))
            logits_t = 0.0
            for j in tl.static_range(0, 128):
                logits_t += q_sub[j] * k_vec_t[j]
            total_exp += tl.exp((logits_t * sm_scale))
        prob_i = tl.exp(logits_scaled_i) / total_exp

        # Accumulate output: out_vec += prob_i * v_vec
        for j in tl.static_range(0, 128):
            out_vec[j] += prob_i * v_vec[j]

    # Store output for this (b, q_idx, h)
    out_off = global_q_idx * (32 * 128) + h * 128
    for j in tl.static_range(0, 128):
        tl.store(out_ptr + out_off + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.GQA_RATIO = self.num_qo_heads // self.num_kv_heads  # 4
        self.MAX_KV = 256  # covers typical candidate_max values safely

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        # Cast to float32 for compute; keep original dtype for output
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # k_cache and v_cache have shape [num_pages, 1, 8, 128] in original, squeeze(1) to [num_pages, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        len_indptr = qo_indptr.shape[0]
        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        # Original asserts
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert self.num_kv_heads == 8 and head_dim == 128, "num_kv_heads must be 8 and head_dim must be 128"

        # Allocate outputs (float32 for compute; will cast to bfloat16 if needed)
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, q_idx, h)
        grid = (len_indptr, total_q, num_qo_heads)
        _compute_one_triplet_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices,
            out_f32, lse_f32,
            GQA_RATIO=self.GQA_RATIO,
            MAX_KV=self.MAX_KV,
            sm_scale=float(sm_scale),
            num_warps=4,
            num_stages=2,
        )

        # Return output as bfloat16 to match original; lse as float32
        output_bf16 = out_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
