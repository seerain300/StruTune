import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,               # *fp32, [total_q * num_qo_heads, head_dim]
    k_ptr,               # *fp32, [num_pages * num_kv_heads, head_dim]
    v_ptr,               # *fp32, [num_pages * num_kv_heads, head_dim]
    qo_indptr,           # *int32, [len_indptr]
    kv_indptr,           # *int32, [len_indptr]
    kv_indices,          # *int32, [num_kv_indices]
    output_ptr,          # *fp32, [total_q * num_qo_heads, head_dim]
    output_lse_ptr,      # *fp32, [total_q * num_qo_heads]
    sm_scale,            # fp32 scalar
    q_start,             # int32
    q_end,               # int32
    kv_start,            # int32
    kv_end,              # int32
    total_q,             # int32
    num_qo_heads,        # int32, e.g., 32
    num_kv_heads,        # int32, e.g., 8
    head_dim,            # int32, e.g., 128
    MAX_Q_SEG: tl.constexpr,   # e.g., 128
    MAX_KV_SEG: tl.constexpr,  # e.g., 256
):
    # One program per segment
    b = tl.program_id(0)

    # Compute segment bounds from input indptrs (host passes q_start/q_end already)
    # num_q_tokens_segment = q_end - q_start
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Constants
    ln2 = 0.6931471805599453  # math.log(2.0)

    # Iterate over query tokens in segment using static loop + mask
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            # Mask out in stores/loads if needed; we'll continue for structure
            continue
        global_q_idx = q_start + q_i
        row_offset = global_q_idx * num_qo_heads

        # For each query head h in [0, 32)
        for h in range(0, 32):
            kv_head = h // (num_qo_heads // num_kv_heads)  # GQA ratio = 4

            # First pass: compute max and sum_exp for logsumexp
            max_val = -float('inf')
            sum_exp = 0.0

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
                # Update logsumexp
                if logits > max_val:
                    # Reinitialize sum_exp when new max is found
                    sum_exp = 0.0
                    for j in range(0, MAX_KV_SEG):
                        if j >= num_kv_tokens:
                            break
                        k_j_idx = kv_indices[kv_start + j]
                        q_j = tl.load(q_ptr + (q_start + j) * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim))  # [head_dim]
                        k_j_base = k_j_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                        k_j = tl.load(k_ptr + k_j_base + tl.arange(0, head_dim))  # [head_dim]
                        dot_j = tl.sum(q_j * k_j, axis=0)  # scalar
                        log_j = dot_j * sm_scale
                        sum_exp += tl.exp(log_j - max_val)
                    max_val = logits

            # Compute lse for this (q_i, h)
            lse_val = (max_val + tl.log(sum_exp)) * (1.0 / ln2)
            # Store lse into output_lse_ptr at row_offset * num_qo_heads + h
            tl.store(output_lse_ptr + row_offset * num_qo_heads + h, lse_val)

            # Second pass: compute attention and output vector
            max_val2 = -float('inf')
            sum_exp2 = 0.0
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = kv_indices[kv_start + kk]
                q_vec = tl.load(q_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim))  # [head_dim]
                k_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim))  # [head_dim]
                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                logits = dot * sm_scale  # scalar
                if logits > max_val2:
                    sum_exp2 = 0.0
                    for j in range(0, MAX_KV_SEG):
                        if j >= num_kv_tokens:
                            break
                        k_j_idx = kv_indices[kv_start + j]
                        q_j = tl.load(q_ptr + (q_start + j) * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim))  # [head_dim]
                        k_j_base = k_j_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                        k_j = tl.load(k_ptr + k_j_base + tl.arange(0, head_dim))  # [head_dim]
                        dot_j = tl.sum(q_j * k_j, axis=0)  # scalar
                        log_j = dot_j * sm_scale
                        sum_exp2 += tl.exp(log_j - max_val2)
                    max_val2 = logits

                attn = tl.exp((dot * sm_scale) - max_val2)  # softmax contribution

                v_base = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                v_vec = tl.load(v_ptr + v_base + tl.arange(0, head_dim))  # [head_dim]
                out_vec += attn * v_vec

            # Store output vector for head h at (global_q_idx)
            tl.store(output_ptr + row_offset * head_dim + h * head_dim + tl.arange(0, head_dim), out_vec, mask=True)


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

        # Precompute per-segment bounds for launch
        # len_indptr must be at least 2 for a valid segment
        assert qo_indptr.shape[0] >= 2 and kv_indptr.shape[0] >= 2, "indptr length must be >= 2"
        q_start = int(qo_indptr[0].item())
        q_end = int(qo_indptr[1].item())
        kv_start = int(kv_indptr[0].item())
        kv_end = int(kv_indptr[1].item())

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)

        attention_kernel[grid](
            q_ptr=q.view(total_q * num_qo_heads, head_dim),
            k_ptr=k_cache_flat,
            v_ptr=v_cache_flat,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            output_ptr=output.view(total_q * num_qo_heads, head_dim),
            output_lse_ptr=output_lse.view(total_q * num_qo_heads),
            sm_scale=float(sm_scale),
            q_start=q_start,
            q_end=q_end,
            kv_start=kv_start,
            kv_end=kv_end,
            total_q=total_q,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            MAX_Q_SEG=128,
            MAX_KV_SEG=256,
            num_warps=4,
            num_stages=2,
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
