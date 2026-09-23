import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128]
    v_ptr,          # *f32, [num_pages, 8, 128]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    kv_indices_ptr, # *i32, [num_kv_indices]
    output_lse_ptr, # *f32, [total_q, 32]
    sm_scale,       # f32 scalar
    len_indptr,     # i32
    total_q,        # i32
    head_dim,       # i32, 128
    num_qo_heads,   # i32, 32
    num_kv_heads,   # i32, 8
    gqa_ratio,      # i32, 4
    ln2_inv,        # f32, 1 / ln(2)
    MAX_Q_SEG: tl.constexpr,   # e.g., 4096
    MAX_KV_SEG: tl.constexpr,  # e.g., 4096
):
    # One program per segment
    b = tl.program_id(0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Loop over query positions
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            continue
        global_q_idx = q_start + q_i

        # Loop over query heads
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio
            # Load q vector for this head
            q_vec = tl.load(q_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim))

            # Compute logsumexp over key list
            max_val = -float('inf')
            sum_exp = 0.0

            # Iterate keys up to MAX_KV_SEG with mask
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim))
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                # Update logsumexp
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)


@triton.jit
def compute_output_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128]
    v_ptr,          # *f32, [num_pages, 8, 128]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    kv_indices_ptr, # *i32, [num_kv_indices]
    output_ptr,     # *f32, [total_q, 32, 128]
    sm_scale,       # f32 scalar
    len_indptr,     # i32
    total_q,        # i32
    head_dim,       # i32, 128
    num_qo_heads,   # i32, 32
    num_kv_heads,   # i32, 8
    gqa_ratio,      # i32, 4
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
):
    b = tl.program_id(0)
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            continue
        global_q_idx = q_start + q_i
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio
            q_vec = tl.load(q_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim))

            logits = tl.zeros([MAX_KV_SEG], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim))
                prod = tl.sum(q_vec * k_vec, axis=0)
                logits[kk] = prod * sm_scale

            # Causal mask: max_kv_idx = min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
            max_kv_idx = tl.minimum(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)

            # Compute max and sum_exp
            max_val = -float('inf')
            for kk in range(0, MAX_KV_SEG):
                if kk < max_kv_idx:
                    max_val = tl.maximum(max_val, logits[kk])

            sum_exp = 0.0
            for kk in range(0, MAX_KV_SEG):
                if kk < max_kv_idx:
                    sum_exp += tl.exp(logits[kk] - max_val)

            softmax_vals = tl.zeros([MAX_KV_SEG], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk < max_kv_idx:
                    softmax_vals[kk] = tl.exp(logits[kk] - max_val) / sum_exp
                else:
                    softmax_vals[kk] = 0.0

            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk < max_kv_idx:
                    v_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                    v_vec = tl.load(v_ptr + v_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim))
                    out_vec += softmax_vals[kk] * v_vec

            tl.store(output_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Output buffers (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Constants
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2_inv = 1.0 / math.log(2.0)

        MAX_Q_SEG = 4096
        MAX_KV_SEG = 4096

        # Launch Triton kernels: one program per segment
        compute_lse_kernel[(len_indptr - 1,)](
            q, k_cache, v_cache,
            qo_indptr, kv_indptr, kv_indices,
            lse,
            sm_scale, len_indptr, total_q, head_dim, num_qo_heads, num_kv_heads, gqa_ratio, ln2_inv,
            MAX_Q_SEG=MAX_Q_SEG, MAX_KV_SEG=MAX_KV_SEG,
            num_warps=4, num_stages=2,
        )

        compute_output_kernel[(len_indptr - 1,)](
            q, k_cache, v_cache,
            qo_indptr, kv_indptr, kv_indices,
            output,
            sm_scale, len_indptr, total_q, head_dim, num_qo_heads, num_kv_heads, gqa_ratio,
            MAX_Q_SEG=MAX_Q_SEG, MAX_KV_SEG=MAX_KV_SEG,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
