import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_and_out_per_triplet(
    q_ptr,            # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr, v_ptr,     # *float32, [num_pages, num_kv_heads, head_dim] (3D)
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,          # *float32, [total_q, num_qo_heads]
    sm_scale,         # float32 scalar
    GQA_RATIO: tl.constexpr,   # e.g., 4
    MAX_KV: tl.constexpr,      # e.g., 256
    HEAD_DIM: tl.constexpr,    # e.g., 128
):
    # Grid is (len_indptr, total_q, num_qo_heads)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load segment starts
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and KV indices in this batch
    num_q_tokens_in_b = qo_end - qo_start
    num_kv_indices_in_b = kv_end - kv_start

    # Causal + length masking: max valid i is candidate_max and must be < num_kv_indices_in_b and in-range
    candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens_in_b)
    kv_head = h // GQA_RATIO

    # Initialize accumulators
    # We'll build logits vector of length MAX_KV by masking invalid entries
    logits = tl.zeros((MAX_KV,), dtype=tl.float32)

    # Loop over candidate keys with mask (avoid dynamic while / returns)
    for i in tl.static_range(0, MAX_KV):
        # Check bounds: i < candidate_max, i < num_kv_indices_in_b, and kv_start + i < kv_end
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & ((kv_start + i) < kv_end)
        idx = tl.load(kv_indices_ptr + kv_start + i, mask=valid_i, other=0)
        # Compute flat pointer offsets for k_ptr and v_ptr:
        # shape is [num_pages, num_kv_heads, head_dim], so offset = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        # We don't know num_pages here, but we can still load using idx and kv_head:
        # Note: idx indexes the batch of kv_indices within this batch b; to access the specific cache,
        # we rely on the fact that k_ptr and v_ptr are laid out as [num_pages, 8, 128], and idx is in [0, num_pages).
        # The kernel is launched per b, so we can assume idx < num_pages for this batch b.
        k_off = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
        v_off = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM

        # Load k_sub and v_sub vectors; mask invalid_i to zero
        k_sub = tl.load(k_ptr + k_off, mask=valid_i, other=0.0)  # [HEAD_DIM]
        v_sub = tl.load(v_ptr + v_off, mask=valid_i, other=0.0)  # [HEAD_DIM]

        # Load q_sub vector: q[global_q_idx, h, :]
        global_q_idx = qo_start + q_idx
        q_off = global_q_idx * (32 * HEAD_DIM) + h * HEAD_DIM
        q_sub = tl.load(q_ptr + q_off)  # scalar vector load across HEAD_DIM implied by static loop? Not correct.

        # We need q_sub as a 1D vector of length HEAD_DIM; Triton does not support dynamic vector loads across HEAD_DIM in a loop.
        # Instead, load q_sub components j in a static range and accumulate dot with k_sub:
        # But Triton requires static shapes; better approach: load q_sub as a single vector before loop by j in range.
        # We'll fix q_sub loading outside the i-loop.

    # Correct approach: load q_sub once, then accumulate logits over i
    # Load q_sub once
    q_off_q = (qo_start + q_idx) * (32 * HEAD_DIM) + h * HEAD_DIM
    q_sub = tl.load(q_ptr + q_off_q)  # This loads a [HEAD_DIM] vector? Triton requires explicit indexing; we need to build a vector.

    # We cannot directly load a [HEAD_DIM] vector with Triton's pointer-based load in a concise way; instead, we compute dot per i:
    # We'll re-load q_sub components inside i-loop; Triton supports scalar loads; accumulate into logits.

    # Reinitialize logits
    logits = tl.zeros((MAX_KV,), dtype=tl.float32)

    # Loop again and accumulate dot products for logits
    for i in tl.static_range(0, MAX_KV):
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & ((kv_start + i) < kv_end)
        idx = tl.load(kv_indices_ptr + kv_start + i, mask=valid_i, other=0)
        k_off = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
        v_off = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
        k_sub = tl.load(k_ptr + k_off, mask=valid_i, other=0.0)  # [HEAD_DIM]
        v_sub = tl.load(v_ptr + v_off, mask=valid_i, other=0.0)  # [HEAD_DIM]

        # q_sub is scalar for this head; we need to load its vector across dimensions. Better: compute dot with per-element loads.
        # Build q_sub vector by loading each j:
        q_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        for j in tl.static_range(0, HEAD_DIM):
            q_elem_off = ((qo_start + q_idx) * (32 * HEAD_DIM) + h * HEAD_DIM + j)
            q_elem = tl.load(q_ptr + q_elem_off)
            q_vec[j] = q_elem
        # Compute dot = sum_j q_vec[j] * k_sub[j]
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_sub[j]
        logits[i] = dot

    # Now compute logsumexp of scaled logits
    logits_scaled = logits * sm_scale
    # lse = logsumexp_base2(logits_scaled) = logsumexp(logits_scaled) / log(2)
    # Triton has tl.log but not tl.logsumexp; we implement it via a loop reduction for MAX_KV which is small
    max_log = -float("inf")
    for i in tl.static_range(0, MAX_KV):
        if valid_i[i]:  # valid_i is Triton bool vector; we can use it to ignore invalid
            max_log = tl.maximum(max_log, logits_scaled[i])
    sum_exp = 0.0
    for i in tl.static_range(0, MAX_KV):
        if valid_i[i]:
            sum_exp += tl.exp(logits_scaled[i] - max_log)
    lse_val = max_log + tl.log(sum_exp)  # natural log; divide by log(2) later
    lse_val = lse_val / math.log(2.0)

    # Compute output vector: out = softmax(logits_scaled) @ v_sub per i
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for i in tl.static_range(0, MAX_KV):
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & ((kv_start + i) < kv_end)
        idx = tl.load(kv_indices_ptr + kv_start + i, mask=valid_i, other=0)
        k_off = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
        v_off = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
        k_sub = tl.load(k_ptr + k_off, mask=valid_i, other=0.0)  # not used here, but kept for consistency
        v_sub = tl.load(v_ptr + v_off, mask=valid_i, other=0.0)  # [HEAD_DIM]
        # prob_i = exp(logits_scaled[i] - lse_val) / sum_exp
        # However, sum_exp is per logsumexp, not per i; instead compute softmax per i for scaling:
        # We need probabilities over valid i. Compute denom across valid_i.
        prob = 0.0
        sum_soft = 0.0
        for j in tl.static_range(0, MAX_KV):
            valid_j = (j < candidate_max) & (j < num_kv_indices_in_b) & ((kv_start + j) < kv_end)
            if valid_j:
                sum_soft += tl.exp(logits_scaled[j] - lse_val)
        # Re-calculate prob_i
        prob_i = tl.exp(logits_scaled[i] - lse_val) / sum_soft
        # Build q_sub for matvec with v_sub: q_sub is scalar head; out contribution per i is prob_i * v_sub
        # But prob_i is scalar; need to multiply with v_sub vector? Here out_vec += prob_i * v_sub (scalar times vector is not supported).
        # Instead, we compute a scalar contribution to out_vec via an inner loop over v_sub components:
        # Compute dot with q_sub: q_sub = load q[global_q_idx, h, :] vector
        # Load q_sub vector across HEAD_DIM
        q_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        for j in tl.static_range(0, HEAD_DIM):
            q_elem_off = ((qo_start + q_idx) * (32 * HEAD_DIM) + h * HEAD_DIM + j)
            q_elem = tl.load(q_ptr + q_elem_off)
            q_vec[j] = q_elem
        # out_vec += prob_i * v_sub
        # Multiply each v_sub[j] by prob_i
        for j in tl.static_range(0, HEAD_DIM):
            out_vec[j] += prob_i * v_sub[j]

    # Store output for this (b, q_idx, h)
    out_off = (qo_start + q_idx) * (32 * HEAD_DIM) + h * HEAD_DIM
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])

    # Store lse for this (b, q_idx, h)
    lse_off = (qo_start + q_idx) * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.GQA_RATIO = self.num_qo_heads // self.num_kv_heads  # 4
        self.MAX_KV = 256  # conservative upper bound for candidate_max

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        # Cast to float32 and make contiguous
        q_f32 = q.to(torch.float32).contiguous()
        # Preprocess k_cache and v_cache: squeeze the 1-sized dimension (as in original)
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        len_indptr = qo_indptr.shape[0]
        total_q = int(qo_indptr[-1].item())  # number of queries overall
        num_qo_heads = self.num_qo_heads
        head_dim = self.head_dim

        # Allocate outputs
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, q_idx, h)
        grid = (len_indptr, total_q, num_qo_heads)
        _compute_lse_and_out_per_triplet[grid](
            q_f32, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices,
            out_f32, lse_f32,
            sm_scale, GQA_RATIO=self.GQA_RATIO, MAX_KV=self.MAX_KV, HEAD_DIM=self.head_dim,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 for final result; lse remains float32
        output_bf16 = out_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
