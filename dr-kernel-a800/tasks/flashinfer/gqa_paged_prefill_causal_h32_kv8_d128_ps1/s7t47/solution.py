import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,              # *fp32, [total_q * num_qo_heads, head_dim]
    k_ptr,              # *fp32, [num_pages * num_kv_heads, head_dim]
    v_ptr,              # *fp32, [num_pages * num_kv_heads, head_dim]
    kv_indices,         # *int32, [num_kv_indices]
    output_ptr,         # *fp32, [total_q * num_qo_heads, head_dim]
    output_lse_ptr,     # *fp32, [total_q * num_qo_heads]
    sm_scale,           # fp32 scalar
    q_start,            # int32
    q_end,              # int32
    kv_start,           # int32
    kv_end,             # int32
    total_q,            # int32
    num_qo_heads,       # int32 (e.g., 32)
    num_kv_heads,       # int32 (e.g., 8)
    head_dim,           # int32 (e.g., 128)
    gqa_ratio,          # int32 (num_qo_heads // num_kv_heads, e.g., 4)
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
):
    # One program per segment
    b = tl.program_id(0)

    # Segment bounds (host passed)
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Precompute constants
    ln2 = 0.6931471805599453  # math.log(2.0)
    ln2_inv = 1.0 / ln2

    # Iterate over query tokens in segment using static loop + mask
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            continue
        global_q_idx = q_start + q_i
        row_offset = global_q_idx * num_qo_heads

        # Process each query head h
        for h in range(0, 32):
            kv_head = h // gqa_ratio  # GQA mapping

            # First pass: compute logsumexp over keys (scaled logits)
            m = -float('inf')  # running max
            sum_exp = 0.0      # running sum in stable form

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = kv_indices[kv_start + kk]  # int32 index

                # Load q[h, :] vector [head_dim]
                q_base = row_offset * head_dim + h * head_dim
                q_vec = tl.load(q_ptr + q_base + tl.arange(0, head_dim))  # [head_dim]

                # Load k[k_idx, kv_head, :] vector [head_dim]
                k_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim))  # [head_dim]

                # Compute dot product
                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar

                # Scaled logits
                logit = dot * sm_scale

                # Stable logsumexp update
                m_new = tl.maximum(m, logit)
                sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(logit - m_new)
                m = m_new

            # Compute lse = log(sum_exp) / ln(2)
            lse_val = (m + tl.log(sum_exp)) * ln2_inv

            # Compute causal max_kv_idx: mask queries beyond num_kv_tokens
            # max_kv_idx = min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
            delta = num_kv_tokens - num_q_tokens_segment
            max_kv_idx = q_i + 1 + delta
            max_kv_idx = tl.minimum(max_kv_idx, num_kv_tokens)

            # Second pass: compute softmax attention up to max_kv_idx and accumulate total
            total = 0.0  # avoid division by zero

            for kk in range(0, MAX_KV_SEG):
                if kk >= max_kv_idx:
                    break
                k_idx = kv_indices[kv_start + kk]  # int32 index

                q_base = row_offset * head_dim + h * head_dim
                q_vec = tl.load(q_ptr + q_base + tl.arange(0, head_dim))  # [head_dim]

                k_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim))  # [head_dim]

                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                logit = dot * sm_scale

                attn_k = tl.exp(logit - lse_val)
                total += attn_k

            # Clamp total to avoid division by zero
            total = tl.maximum(total, 1e-20)

            # Third pass: accumulate output vector for head h
            for kk in range(0, MAX_KV_SEG):
                if kk >= max_kv_idx:
                    break
                k_idx = kv_indices[kv_start + kk]  # int32 index

                q_base = row_offset * head_dim + h * head_dim
                q_vec = tl.load(q_ptr + q_base + tl.arange(0, head_dim))  # [head_dim]

                k_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim))  # [head_dim]

                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                logit = dot * sm_scale

                attn_k = tl.exp(logit - lse_val)

                v_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                v_vec = tl.load(v_ptr + v_base + tl.arange(0, head_dim))  # [head_dim]

                out_vec = attn_k * v_vec  # elementwise product

                # Store output vector for head h at (global_q_idx, h)
                tl.store(output_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim), out_vec)

            # Store lse for (global_q_idx, h)
            tl.store(output_lse_ptr + (global_q_idx * num_qo_heads + h), lse_val)


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

        # Flatten k_cache and v_cache to [num_pages, num_kv_heads*head_dim]
        k_cache_flat = k_cache.view(num_pages, num_kv_heads * head_dim)
        v_cache_flat = v_cache.view(num_pages, num_kv_heads * head_dim)

        # Output buffers (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        output_lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)

        attention_kernel[grid](
            q_ptr=q.view(total_q * num_qo_heads, head_dim),              # q_ptr: [total_q*32, 128]
            k_ptr=k_cache_flat,                                           # k_ptr: [num_pages, 8*128]
            v_ptr=v_cache_flat,                                           # v_ptr: [num_pages, 8*128]
            kv_indices=kv_indices,                                       # [num_kv_indices]
            output_ptr=output.view(total_q * num_qo_heads, head_dim),    # [total_q*32, 128]
            output_lse_ptr=output_lse,                                   # [total_q*32]
            sm_scale=float(sm_scale),                                    # fp32 scalar
            q_start=qo_indptr[0].item(),
            q_end=qo_indptr[1].item(),
            kv_start=kv_indptr[0].item(),
            kv_end=kv_indptr[1].item(),
            total_q=total_q,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            gqa_ratio=(num_qo_heads // num_kv_heads),                    # 4
            MAX_Q_SEG=256,                                               # conservative upper bound
            MAX_KV_SEG=512,                                              # conservative upper bound
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
