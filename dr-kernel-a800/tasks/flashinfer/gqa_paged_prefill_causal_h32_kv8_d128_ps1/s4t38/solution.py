import torch
import math
import triton
import triton.language as tl


@triton.jit
def attn_gqa_token_kernel(
    q_ptr,         # *float32, [total_q, NUM_QO_HEADS, HEAD_DIM]
    k_ptr,         # *float32, [num_kv_tokens * NUM_KV_HEADS, HEAD_DIM] but indexed by kv_indices and kv_head
    v_ptr,         # *float32, same shape as k_ptr
    lse_ptr,       # *float32, [len_indptr - 1, NUM_QO_HEADS]
    out_ptr,       # *float32, [total_q, NUM_QO_HEADS, HEAD_DIM]
    qo_indptr_ptr, # *i32, [len_indptr]
    kv_indptr_ptr, # *i32, [len_indptr]
    kv_indices_ptr,# *i32, [num_kv_indices]
    NUM_QO_HEADS: tl.constexpr,     # 32
    NUM_KV_HEADS: tl.constexpr,     # 8
    HEAD_DIM: tl.constexpr,         # 128
    GQA_RATIO: tl.constexpr,        # 4
    SM_SCALE: tl.constexpr,         # float32, e.g., 1.0 / sqrt(128)
):
    # Grid: (len_indptr - 1, total_q)
    b = tl.program_id(0)            # batch element index
    token_global = tl.program_id(1) # global token index across all batches

    q_start = tl.load(qo_indptr_ptr + b)       # i32
    q_end = tl.load(qo_indptr_ptr + b + 1)     # i32
    kv_start = tl.load(kv_indptr_ptr + b)      # i32
    kv_end = tl.load(kv_indptr_ptr + b + 1)    # i32

    if q_start >= q_end or kv_start >= kv_end:
        return

    local_q_idx = token_global - q_start
    if local_q_idx < 0 or local_q_idx >= (q_end - q_start):
        return

    # For this (b, local_q_idx), compute lse and output per query head h
    for h in range(NUM_QO_HEADS):
        # Initialize lse components
        max_logit = -float("inf")
        sum_exp = 0.0

        # Pass 1: compute logsumexp across kv tokens (only need max and sum_exp)
        for i in range(kv_start, kv_end):
            kv_index = tl.load(kv_indices_ptr + i)  # int32
            kv_head = h // GQA_RATIO

            # Load q vector element h, convert to float32
            row_q = local_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
            q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(HEAD_DIM):
                q_elem = tl.load(q_ptr + row_q + d)  # float32 (q already float32)
                q_vec[d] = q_elem

            # Load k vector for this kv token and head
            idx_k = (kv_index * NUM_KV_HEADS + kv_head) * HEAD_DIM
            k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(HEAD_DIM):
                k_elem = tl.load(k_ptr + idx_k + d)  # float32
                k_vec[d] = k_elem

            dot = 0.0
            for d in range(HEAD_DIM):
                dot += q_vec[d] * k_vec[d]
            logits_scaled = dot * SM_SCALE

            # Update max and sum_exp
            if logits_scaled > max_logit:
                max_logit = logits_scaled
            sum_exp += tl.exp(logits_scaled - max_logit)

        # Store max_logit and sum_exp for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, max_logit)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h + NUM_QO_HEADS, sum_exp)

        # Second pass: compute softmax and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(kv_start, kv_end):
            kv_index = tl.load(kv_indices_ptr + i)  # int32
            kv_head = h // GQA_RATIO

            # Load q vector element h
            row_q = local_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
            q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(HEAD_DIM):
                q_elem = tl.load(q_ptr + row_q + d)
                q_vec[d] = q_elem

            # Load k vector for this kv token and head
            idx_k = (kv_index * NUM_KV_HEADS + kv_head) * HEAD_DIM
            k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(HEAD_DIM):
                k_elem = tl.load(k_ptr + idx_k + d)
                k_vec[d] = k_elem

            dot = 0.0
            for d in range(HEAD_DIM):
                dot += q_vec[d] * k_vec[d]
            logits_scaled = dot * SM_SCALE

            # Softmax component using stored max and sum
            exp_val = tl.exp(logits_scaled - max_logit)   # numerator
            soft = exp_val / sum_exp                      # denominator is sum_exp

            # Load v vector for this kv token and head
            idx_v = (kv_index * NUM_KV_HEADS + kv_head) * HEAD_DIM
            v_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(HEAD_DIM):
                v_elem = tl.load(v_ptr + idx_v + d)
                v_vec[d] = v_elem

            out_vec += soft * v_vec

        # Store output vector to out_ptr at row = token_global * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        row_out = token_global * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        for d in range(HEAD_DIM):
            tl.store(out_ptr + row_out + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q.device
        # Compute in float32, return output as bfloat16 (to match original)
        q_f32 = q.to(torch.float32)
        # Squeeze "pages" dimension
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        # Ensure inputs are contiguous
        q_f32 = q_f32.contiguous()
        k_cache_flat = k_cache_flat.contiguous()
        v_cache_flat = v_cache_flat.contiguous()

        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        len_indptr = qo_indptr.shape[0]
        total_q = q.shape[0]

        # Allocate outputs
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((len_indptr - 1, num_qo_heads), dtype=torch.float32, device=device)

        # Launch fused Triton kernel: one program per (b, token_global)
        grid = (len_indptr - 1, total_q)
        attn_gqa_token_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat, lse, out, qo_indptr, kv_indptr, kv_indices,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4, SM_SCALE=sm_scale,
            num_warps=1, num_stages=1
        )

        # Return output cast to bfloat16 and lse
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
