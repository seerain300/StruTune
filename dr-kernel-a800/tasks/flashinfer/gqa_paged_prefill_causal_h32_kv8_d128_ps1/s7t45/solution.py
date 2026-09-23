import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,              # [total_q * 32, 128] float32
    k_ptr,              # [num_pages, 8*128]   float32
    v_ptr,              # [num_pages, 8*128]   float32
    qo_indptr,          # [len_indptr] int32
    kv_indptr,          # [len_indptr] int32
    kv_indices,         # [num_kv_indices] int32
    output_ptr,         # [total_q * 32, 128] float32
    output_lse_ptr,     # [total_q * 32]     float32
    sm_scale,           # float32
    q_start,            # int32
    q_end,              # int32
    kv_start,           # int32
    kv_end,             # int32
    head_dim: tl.constexpr,          # 128
    num_qo_heads: tl.constexpr,      # 32
    num_kv_heads: tl.constexpr,      # 8
    gqa_ratio: tl.constexpr,         # 4
    MAX_Q_SEG: tl.constexpr,         # upper bound (e.g., 128)
    MAX_KV_SEG: tl.constexpr,        # upper bound (e.g., 128)
):
    # One program per segment b in [0, len_indptr-2]
    b = tl.program_id(0)

    # Segment bounds (scalars passed from host)
    num_q_tokens_segment = q_end - q_start  # int32
    num_kv_tokens = kv_end - kv_start       # int32

    # Iterate over query tokens in segment with static loop and mask
    for q_i in range(0, MAX_Q_SEG):
        # Mask for valid query token
        valid_q = q_i < num_q_tokens_segment
        if valid_q:
            global_q_idx = q_start + q_i
            row_offset = global_q_idx * num_qo_heads

            # For each query head h
            for h in range(0, 32):
                kv_head = h // gqa_ratio  # 4

                # Compute logsumexp over keys (numerically stable)
                max_val = -float("inf")
                sum_exp = 0.0

                for kk in range(0, MAX_KV_SEG):
                    valid_k = kk < num_kv_tokens
                    if valid_k:
                        k_idx = kv_indices[kv_start + kk]  # int32
                        # Load q vector [head_dim]
                        q_vec = tl.load(q_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
                        # Load k vector [head_dim] for this kv_idx and kv_head
                        k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
                        # Dot product
                        dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                        # Scaled logits
                        logits = dot * sm_scale
                        # Stable lse accumulation
                        if max_val < logits:
                            sum_exp = sum_exp * tl.exp(max_val - logits) + 1.0
                            max_val = logits
                        else:
                            sum_exp = sum_exp + tl.exp(logits - max_val)

                lse_val = (max_val + tl.log(sum_exp)) * (1.0 / 0.6931471805599453)  # ln(2)^{-1}
                # Store LSE for (global_q_idx, h)
                tl.store(output_lse_ptr + row_offset + h, lse_val)

                # Compute attention and output
                # attn over keys up to causal cutoff: max_kv_idx = min(q_i + 1, num_kv_tokens)
                max_kv_idx = tl.minimum(q_i + 1, num_kv_tokens)
                final_out_vec = tl.zeros([head_dim], dtype=tl.float32)

                for kk in range(0, MAX_KV_SEG):
                    valid_k = kk < max_kv_idx
                    if valid_k:
                        k_idx = kv_indices[kv_start + kk]
                        k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
                        q_vec = tl.load(q_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
                        dot = tl.sum(q_vec * k_vec, axis=0)
                        logits = dot * sm_scale
                        attn_k = tl.exp(logits)  # softmax contribution
                        v_vec = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
                        final_out_vec += attn_k * v_vec

                # Store output vector for head h at (global_q_idx)
                tl.store(output_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim), final_out_vec, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes assertions
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Ensure contiguity and float32
        q = q.contiguous().to(torch.float32)  # [total_q, 32, 128]
        k_cache = k_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]
        v_cache = v_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]

        # Flatten k_cache and v_cache to [num_pages, 8*128]
        num_kv_heads = k_cache.shape[2]
        k_cache_flat = k_cache.view(num_pages, num_kv_heads * head_dim)
        v_cache_flat = v_cache.view(num_pages, num_kv_heads * head_dim)

        # Output buffers (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        output_lse = torch.empty


def run(*args):
    return ModelNew()(*args)
