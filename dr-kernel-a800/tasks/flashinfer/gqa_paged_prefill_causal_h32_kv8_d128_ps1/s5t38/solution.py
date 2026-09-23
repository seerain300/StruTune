import math
import torch

import triton
import triton.language as tl


@triton.jit
def _batch_attention_kernel(
    q_ptr,            # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,     # *float32, [num_pages, 8, 128] (squeezed from original [num_pages, 1, 8, 128])
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, 32, 128]
    lse_ptr,          # *float32, [total_q, 32]
    GQA_RATIO: tl.constexpr,   # 4
    HEAD_DIM: tl.constexpr,    # 128
    sm_scale: tl.float32,      # scaling factor
):
    # One program per batch b
    b = tl.program_id(0)

    # Load qo_indptr[b] and qo_indptr[b+1]
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)

    # Load kv_indptr[b] and kv_indptr[b+1]
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Precompute candidate max for each query q_idx: candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    # We will loop over q_idx and compute per query.

    # Loop over query indices: static range up to num_q_tokens; Triton supports scalar range here
    for q_idx in range(0, 65535):  # upper bound; we'll guard with num_q_tokens
        if q_idx >= num_q_tokens:
            break
        global_q_idx = qo_start + q_idx

        # Loop over heads h in [0, 32)
        for h in range(0, 32):
            # GQA mapping: query head uses KV head h // 4
            kv_head = h // GQA_RATIO  # 32 // 8 == 4

            # Load q vector: q[global_q_idx, h, :]
            q_vec = tl.load(q_ptr + global_q_idx * (32 * 128) + h * 128 + tl.arange(0, HEAD_DIM))

            # First pass: compute logsumexp over i in [0, num_kv_indices) with causal mask
            m = -float("inf")
            s = 0.0

            # Iterate i from 0 to MAX_KV (here MAX_KV=num_q_tokens + 1, but we keep static and mask)
            # We set MAX_KV to 65535 (arbitrary large). Mask ensures only i < num_kv_indices contributes.
            for i in range(0, 65535):
                # Validity: i < num_kv_indices and within kv_end
                valid_i = (i < num_kv_indices) & (kv_start + i < kv_end)

                # Gather idx = kv_indices[kv_start + i]
                idx_i = tl.load(kv_indices_ptr + kv_start + i, mask=valid_i, other=0)

                # Compute offsets for k and v rows: k[idx_i, kv_head, :], v[idx_i, kv_head, :]
                k_row_ptr = k_ptr + idx_i * (8 * 128) + kv_head * 128
                v_row_ptr = v_ptr + idx_i * (8 * 128) + kv_head * 128

                k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid_i, other=0.0)
                v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=valid_i, other=0.0)

                # Compute dot product q_sub[h] @ k_vec -> scalar
                dot = 0.0
                for j in tl.static_range(0, HEAD_DIM):
                    dot += q_vec[j] * k_vec[j]
                logits_scaled = dot * sm_scale

                # Update running max and sum for logsumexp
                m_new = tl.maximum(m, logits_scaled)
                s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(valid_i, tl.exp(logits_scaled - m_new), 0.0)
                m = m_new

            # Compute lse
            lse_val = tl.log(s) + m  # natural logsumexp

            # Store lse for this (b, q_idx, h)
            lse_off = global_q_idx * 32 + h
            tl.store(lse_ptr + lse_off, lse_val)

            # Second pass: compute output vector out[h] = sum_i prob_i * v_vec
            out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
            total_sum = 0.0
            for i in range(0, 65535):
                valid_i = (i < num_kv_indices) & (kv_start + i < kv_end)
                idx_i = tl.load(kv_indices_ptr + kv_start + i, mask=valid_i, other=0)

                k_row_ptr = k_ptr + idx_i * (8 * 128) + kv_head * 128
                v_row_ptr = v_ptr + idx_i * (8 * 128) + kv_head * 128

                k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid_i, other=0.0)
                v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=valid_i, other=0.0)

                dot = 0.0
                for j in tl.static_range(0, HEAD_DIM):
                    dot += q_vec[j] * k_vec[j]
                logits_scaled = dot * sm_scale
                prob = tl.exp(logits_scaled - lse_val) * tl.where(valid_i, 1.0, 0.0)
                out_vec += prob * v_vec

            # Store output vector for this (b, q_idx, h)
            out_off = global_q_idx * (32 * 128) + h * 128
            for j in tl.static_range(0, HEAD_DIM):
                tl.store(out_ptr + out_off + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.GQA_RATIO = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        # Cast to float32 for computation; keep original q dtype for output casting
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # Squeeze the (1,) dimension: k_cache has shape [num_pages, 1, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]
        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        assert num_qo_heads == 32 and head_dim == 128, "Expect q shape [*, 32, 128]"

        # Output and lse tensors
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernels: one program per batch b
        grid = (len_indptr,)
        _batch_attention_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices, out_f32, lse_f32,
            GQA_RATIO=self.GQA_RATIO,
            HEAD_DIM=self.head_dim,
            sm_scale=float(sm_scale),
            num_warps=4, num_stages=2
        )

        # Return output in bfloat16 and lse in float32
        output_bf16 = out_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
