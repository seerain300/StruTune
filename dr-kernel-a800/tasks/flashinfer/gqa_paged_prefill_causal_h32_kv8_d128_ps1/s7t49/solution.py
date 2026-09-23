import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,              # *float32, [total_q * num_qo_heads, head_dim] flattened
    k_ptr,              # *float32, [num_pages * num_kv_heads, head_dim] flattened
    v_ptr,              # *float32, [num_pages * num_kv_heads, head_dim] flattened
    qo_indptr,          # *int32, [len_indptr]
    kv_indptr,          # *int32, [len_indptr]
    kv_indices,         # *int32, [num_kv_indices]
    output_ptr,         # *float32, [total_q * num_qo_heads, head_dim] flattened
    output_lse_ptr,     # *float32, [total_q * num_qo_heads]
    sm_scale,           # float32 scalar
    q_start,            # int32
    q_end,              # int32
    kv_start,           # int32
    kv_end,             # int32
    total_q,            # int32
    num_qo_heads,       # int32 (e.g., 32)
    num_kv_heads,       # int32 (e.g., 8)
    head_dim,           # int32 (e.g., 128)
    MAX_Q_SEG: tl.constexpr,   # e.g., 128
    MAX_KV_SEG: tl.constexpr,  # e.g., 128
):
    # One program per segment
    b = tl.program_id(0)

    # Segment bounds (host passed)
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Precompute GQA ratio and ln(2) inverse
    gqa_ratio = num_qo_heads // num_kv_heads  # 4
    ln2_inv = 1.0 / 0.6931471805599453  # 1 / ln(2)

    # Iterate over query tokens in segment using static loop + mask
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            continue
        global_q_idx = q_start + q_i
        row_offset = global_q_idx * num_qo_heads

        # Process each query head h
        for h in range(0, 32):
            kv_head = h // gqa_ratio  # GQA mapping

            # First pass: compute max and sum_exp for logsumexp (scaled logits)
            m = -float('inf')  # max of logits_scaled
            sum_exp = 0.0  # sum of exp(logits_scaled - m)

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = kv_indices[kv_start + kk]  # int32 index

                # Load q[h, :] vector [head_dim]
                q_vec = tl.load(q_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim))  # [head_dim]
                # Load k[k_idx, kv_head, :] vector [head_dim]
                k_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim))  # [head_dim]

                # Compute dot product
                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                logits = dot * sm_scale  # scalar
                m_new = tl.maximum(m, logits)
                sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(logits - m_new)
                m = m_new

            # lse_val = (m + log(sum_exp)) / ln(2)
            lse_val = (m + tl.log(sum_exp)) * ln2_inv

            # Second pass: compute attention and output
            final_out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = kv_indices[kv_start + kk]  # int32 index

                q_vec = tl.load(q_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim))  # [head_dim]
                k_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim))  # [head_dim]

                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                logits = dot * sm_scale  # scalar
                attn = tl.exp(logits - lse_val)  # softmax contribution

                v_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                v_vec = tl.load(v_ptr + v_base + tl.arange(0, head_dim))  # [head_dim]
                final_out_vec += attn * v_vec

            # Store output vector for head h at (global_q_idx)
            tl.store(output_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim), final_out_vec)


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
        output_lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)
        attention_kernel[grid](
            q, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output.view(-1, head_dim), output_lse,
            float(sm_scale),
            int(qo_indptr[b].item()), int(qo_indptr[b + 1].item()),
            int(kv_indptr[b].item()), int(kv_indptr[b + 1].item()),
            int(total_q), int(num_qo_heads), int(num_kv_heads), int(head_dim),
            MAX_Q_SEG=128, MAX_KV_SEG=128,
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
