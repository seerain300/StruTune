import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,               # *f32, [total_q, 32, 128]
    k_ptr,               # *f32, [num_pages, 8, 128]
    v_ptr,               # *f32, [num_pages, 8, 128]
    qo_indptr_ptr,       # *i32, [len_indptr]
    kv_indptr_ptr,       # *i32, [len_indptr]
    kv_indices_ptr,      # *i32, [num_kv_indices]
    output_ptr,          # *f32, [total_q, 32, 128]
    output_lse_ptr,      # *f32, [total_q, 32]
    sm_scale,            # f32 scalar
    len_indptr,          # i32
    total_q,             # i32
    num_qo_heads,        # i32 (32)
    head_dim,            # i32 (128)
    num_kv_heads,        # i32 (8)
    gqa_ratio,           # i32 (4)
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
):
    b = tl.program_id(0)
    # Load segment bounds (scalars per program)
    q_start = tl.load(qo_indptr_ptr + b)
    q_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Constants
    ln2 = 0.6931471805599453
    ln2_inv = 1.0 / ln2

    for q_i in range(0, MAX_Q_SEG):
        # Mask for valid query positions in this segment
        q_valid = q_i < num_q_tokens_segment
        global_q_idx = q_start + q_i
        # If not valid, skip
        if not q_valid:
            continue

        # Process each query head
        for h in range(0, 32):
            # GQA mapping: kv_head = h // 4
            kv_head = h // 4

            # Compute LSE: logsumexp over key list
            max_val = -float('inf')
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                # k_vec: [head_dim]
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim))
                # q_vec: [head_dim]
                q_vec = tl.load(q_ptr + (global_q_idx * (num_qo_heads * head_dim) + h * head_dim + tl.arange(0, head_dim)))
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                # Update logsumexp stably
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * 32 + h, lse_val)

            # Compute output: attn = softmax(logits_scaled), then out = attn @ v
            # For softmax, we need max_kv_idx per causal mask: min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
            max_kv_idx = q_i + 1 + (num_kv_tokens - num_q_tokens_segment)
            if max_kv_idx > num_kv_tokens:
                max_kv_idx = num_kv_tokens
            else:
                max_kv_idx = max(0, max_kv_idx)

            # Build attn vector of size MAX_KV_SEG, mask kk < max_kv_idx
            attn = tl.zeros([MAX_KV_SEG], dtype=tl.float32)
            sum_attn = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= max_kv_idx:
                    continue
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim))
                q_vec = tl.load(q_ptr + (global_q_idx * (num_qo_heads * head_dim) + h * head_dim + tl.arange(0, head_dim)))
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                attn[kk] = tl.exp(scaled)
                sum_attn += attn[kk]

            attn = attn / sum_attn  # ensure sum is 1 (guard for empty attn)

            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= max_kv_idx:
                    continue
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                v_vec = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim))
                out_vec += attn[kk] * v_vec

            # Store output vector for head h
            out_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim + tl.arange(0, head_dim)
            tl.store(output_ptr + out_offset, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        device = q.device
        q = q.contiguous().to(torch.float32)
        k_cache = k_cache.contiguous().to(torch.float32)
        v_cache = v_cache.contiguous().to(torch.float32)
        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        total_q = q.shape[0]
        num_qo_heads = 32
        head_dim = 128
        num_kv_heads = 8
        gqa_ratio = 4  # 32 / 8

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)
        attention_kernel[grid](
            q, k_cache, v_cache,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            sm_scale,
            qo_indptr.shape[0], total_q,
            num_qo_heads, head_dim,
            num_kv_heads, gqa_ratio,
            MAX_Q_SEG=4096, MAX_KV_SEG=4096,
            num_warps=4, num_stages=2,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
