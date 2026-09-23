import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_attention_kernel(
    q_ptr,           # *f32, [total_q, num_qo_heads, head_dim]
    k_ptr,           # *f32, [num_pages, num_kv_heads, head_dim]
    v_ptr,           # *f32, [num_pages, num_kv_heads, head_dim]
    qo_indptr_ptr,   # *i32, [len_indptr]
    kv_indptr_ptr,   # *i32, [len_indptr]
    kv_indices_ptr,  # *i32, [num_kv_indices]
    out_ptr,         # *f32, [total_q, num_qo_heads, head_dim]
    lse_ptr,         # *f32, [total_q, num_qo_heads]
    total_q,         # i32
    num_qo_heads,    # i32 (32)
    head_dim,        # i32 (128)
    num_pages,       # i32
    num_kv_heads,    # i32 (8)
    gqa_ratio,       # i32 (4)
    sm_scale,        # f32
    LN2,             # f32 = log(2)
    MAX_Q: tl.constexpr,             # upper bound on queries per segment (e.g., 4096)
    MAX_KV_TOKENS: tl.constexpr,     # upper bound on kv tokens (e.g., num_pages)
):
    # One program processes one segment (b)
    b = tl.program_id(axis=0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b)         # i32
    q_end = tl.load(qo_indptr_ptr + b + 1)       # i32
    kv_start = tl.load(kv_indptr_ptr + b)        # i32
    kv_end = tl.load(kv_indptr_ptr + b + 1)      # i32

    # Number of valid keys for this segment
    num_kv_tokens = kv_end - kv_start            # i32

    # For each query index within this segment
    for q_idx in range(MAX_Q):
        # If q_idx exceeds actual number of queries in segment, exit
        if q_idx >= (q_end - q_start):
            break
        global_q_idx = q_start + q_idx           # i32

        # Loop over query heads
        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio             # 0..7

            # Compute max allowed keys due to causal masking
            delta = num_kv_tokens - (q_end - q_start)  # i32
            delta = tl.maximum(delta, 0)
            max_kv_idx = q_idx + 1 + delta
            max_kv_idx = tl.minimum(max_kv_idx, num_kv_tokens)  # i32

            # Prepare logits vector for this (segment, q_idx, head)
            logits = tl.zeros((MAX_KV_TOKENS,), dtype=tl.float32)

            # Build k_list and v_list for valid keys in this segment
            # For kk in [0, MAX_KV_TOKENS), valid_k = kk < num_kv_tokens
            # Then kv_idx = kv_start + kk, idx = kv_indices[kv_idx]
            # k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
            for kk in range(MAX_KV_TOKENS):
                valid_k = kk < num_kv_tokens  # scalar bool
                kv_idx = kv_start + kk        # i32
                idx = tl.load(kv_indices_ptr + kv_idx)  # i32
                k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                # Masked load for valid_k
                k_vec = tl.load(k_ptr + k_off, mask=valid_k, other=0.0)  # [head_dim] f32
                v_vec = tl.load(v_ptr + v_off, mask=valid_k, other=0.0)  # [head_dim] f32

                # Load q_vec for head h and accumulate logits
                q_off = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
                q_vec = tl.load(q_ptr + q_off)  # [head_dim] f32

                # Accumulate logits = q_vec @ k_list.T across head_dim
                for d in range(head_dim):
                    logits[kk] += q_vec[d] * k_vec[d]

            # Scale logits
            logits_scaled = logits * sm_scale

            # Apply causal mask: set logits[kk] to -inf for kk >= max_kv_idx
            neg_inf = -float('inf')
            for kk in range(MAX_KV_TOKENS):
                if kk >= max_kv_idx:
                    logits_scaled[kk] = neg_inf

            # Compute LSE = logsumexp(logits_scaled) / LN2
            max_val = -float('inf')
            for kk in range(MAX_KV_TOKENS):
                if logits_scaled[kk] > max_val:
                    max_val = logits_scaled[kk]
            sum_exp = 0.0
            for kk in range(MAX_KV_TOKENS):
                sum_exp += tl.exp(logits_scaled[kk] - max_val)
            lse = max_val + tl.log(sum_exp)  # logsumexp
            lse = lse / LN2  # divide by log(2), matching original

            # Compute softmax attn
            attn = tl.zeros((MAX_KV_TOKENS,), dtype=tl.float32)
            for kk in range(MAX_KV_TOKENS):
                attn[kk] = tl.exp(logits_scaled[kk] - lse)
                if kk >= max_kv_idx:
                    attn[kk] = 0.0

            # Compute out_vec = attn @ v_list
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            for d in range(head_dim):
                # Sum over keys: attn[kk] * v_ptr[kk, kv_head, d]
                for kk in range(MAX_KV_TOKENS):
                    valid_k = kk < num_kv_tokens
                    kv_idx = kv_start + kk
                    idx = tl.load(kv_indices_ptr + kv_idx)
                    v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    v_vec_d = tl.load(v_ptr + v_off + d, mask=valid_k, other=0.0)  # scalar
                    out_vec[d] += attn[kk] * v_vec_d

            # Store output
            out_off = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
            tl.store(out_ptr + out_off, out_vec)

            # Store LSE
            lse_off = global_q_idx * num_qo_heads + h
            tl.store(lse_ptr + lse_off, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity
        device = q.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()

        # Convert to float32 for numerics
        q_f32 = q.to(torch.float32)
        # Flatten the "1" dimension (page_size=1)
        k_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
        v_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

        total_q, num_qo_heads, head_dim = q_f32.shape
        num_pages, num_kv_heads, _ = k_flat.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1
        # Ensure qo_indptr[-1] == total_q
        assert qo_indptr[-1].item() == total_q, "qo_indptr[-1] must equal total_q"

        # Allocate outputs (compute in float32, convert later)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per segment
        grid = (num_segments,)
        LN2 = math.log(2.0)

        # Choose reasonable upper bounds for loops
        MAX_Q = 4096  # upper bound on queries per segment (sufficient for given workloads)
        MAX_KV_TOKENS = num_pages  # upper bound on kv tokens

        compute_attention_kernel[grid](
            q_f32, k_flat, v_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            total_q, num_qo_heads, head_dim,
            num_pages, num_kv_heads, 4,  # gqa_ratio
            sm_scale, LN2,
            MAX_Q=MAX_Q, MAX_KV_TOKENS=MAX_KV_TOKENS,
            num_warps=4, num_stages=2,
        )

        # Return results in original dtypes
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
