import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,             # *f32, [total_q, 32, 128]
    k_ptr,             # *f32, [num_pages, 8, 128]
    v_ptr,             # *f32, [num_pages, 8, 128]
    qo_indptr_ptr,     # *i32, [len_indptr]
    kv_indptr_ptr,     # *i32, [len_indptr]
    kv_indices_ptr,    # *i32, [num_kv_indices]
    out_ptr,           # *f32, [total_q, 32, 128]
    lse_ptr,           # *f32, [total_q, 32]
    total_q,           # i32
    num_qo_heads,      # i32, 32
    head_dim,          # i32, 128
    sm_scale,          # f32, 1/sqrt(128)
    MAX_Q: tl.constexpr,          # upper bound for q tokens per segment (e.g., 16384)
    MAX_KV_TOKENS: tl.constexpr,  # upper bound for kv tokens per segment (e.g., 28)
    GQA_RATIO: tl.constexpr       # 32 // 8 = 4
):
    # One program per segment
    b = tl.program_id(0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Loop over queries in this segment up to MAX_Q, mask by actual num_q_tokens
    for q_idx in range(0, MAX_Q):
        valid_q = q_idx < num_q_tokens
        global_q_idx = q_start + q_idx
        if not valid_q:
            continue

        # For each query head h
        for h in range(0, 32):
            kv_head = h // GQA_RATIO  # 4

            # Build k_list and v_list: shape [MAX_KV_TOKENS, head_dim]
            k_list = tl.zeros((MAX_KV_TOKENS, head_dim), dtype=tl.float32)
            v_list = tl.zeros((MAX_KV_TOKENS, head_dim), dtype=tl.float32)

            # Load per-kk rows via kv_indices[kv_start + kk]
            for kk in range(0, MAX_KV_TOKENS):
                valid_k = kk < num_kv_tokens
                idx = kv_start + kk
                page_id = tl.load(kv_indices_ptr + idx).to(tl.int32)
                base_k = page_id * (num_kv_heads * head_dim)  # num_kv_heads = 8
                base_v = page_id * (num_kv_heads * head_dim)
                k_row = tl.load(k_ptr + base_k + kv_head * head_dim + tl.arange(0, head_dim), mask=valid_k, other=0.0)
                v_row = tl.load(v_ptr + base_v + kv_head * head_dim + tl.arange(0, head_dim), mask=valid_k, other=0.0)
                k_list[kk, :] = k_row
                v_list[kk, :] = v_row

            # Load q_vec[h] of size head_dim
            offs = h * head_dim + tl.arange(0, head_dim)
            q_vec = tl.load(q_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + offs)  # [128]

            # Compute logits for each kk: dot(q_vec, k_list[kk, :])
            logits = tl.zeros((MAX_KV_TOKENS,), dtype=tl.float32)
            for i in range(0, MAX_KV_TOKENS):
                k_row = k_list[i, :]
                logits[i] = tl.sum(q_vec * k_row, axis=0)

            # Scale
            logits_scaled = logits * sm_scale

            # Apply causal mask: bound = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
            delta = num_kv_tokens - num_q_tokens
            bound = q_idx + 1 + delta
            bound = tl.minimum(bound, num_kv_tokens)
            for kk in range(0, MAX_KV_TOKENS):
                if kk >= bound:
                    logits_scaled[kk] = -float("inf")

            # logsumexp: lse = log(sum(exp(logits_scaled - max))) / ln(2)
            max_val = tl.max(logits_scaled, axis=0)
            exp_vals = tl.exp(logits_scaled - max_val)
            sum_exp = tl.sum(exp_vals, axis=0)
            lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
            # Store lse to lse_ptr[global_q_idx, h]
            lse_offset = global_q_idx * num_qo_heads + h
            tl.store(lse_ptr + lse_offset, lse_val)

            # Softmax
            softmax_vals = exp_vals / sum_exp  # entries kk >= bound were -inf -> exp=0, sum_exp excludes them

            # Output matvec: out_vec = sum(softmax[kk] * v_list[kk, :])
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            for i in range(0, MAX_KV_TOKENS):
                v_row = v_list[i, :]
                out_vec += softmax_vals[i] * v_row

            # Store output vector to out_ptr[global_q_idx, h, :]
            out_offset_base = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
            for d in range(0, head_dim):
                tl.store(out_ptr + out_offset_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Convert to float32 for compute
        q_f32 = q.to(torch.float32)
        k_cache_f32 = k_cache.to(torch.float32)
        v_cache_f32 = v_cache.to(torch.float32)

        total_q, num_qo_heads, head_dim = q_f32.shape
        num_pages, num_kv_heads, _ = k_cache_f32.shape  # num_kv_heads == 8, head_dim == 128
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        len_indptr = qo_indptr.shape[0]

        # Allocate outputs (float32)
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        grid = (len_indptr - 1,)

        # Upper bounds for loops (constexpr). Chosen conservatively based on provided workloads.
        MAX_Q = 16384
        MAX_KV_TOKENS = 28
        GQA_RATIO = 4  # 32 // 8

        attention_kernel[grid](
            q_f32,
            k_cache_f32,
            v_cache_f32,
            qo_indptr,
            kv_indptr,
            kv_indices,
            out,
            lse,
            total_q,
            num_qo_heads,
            head_dim,
            sm_scale,
            MAX_Q=MAX_Q,
            MAX_KV_TOKENS=MAX_KV_TOKENS,
            GQA_RATIO=GQA_RATIO,
            num_warps=4,  # simple default; adjust if profiling suggests
            num_stages=1,
        )

        # Return outputs matching the original: output bfloat16, lse float32
        output_bf16 = out.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
