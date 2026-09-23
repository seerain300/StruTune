import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,         # float32 scalar
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,
        half_ln2_inv: tl.constexpr,  # 1 / ln(2)
    ):
        # One program per (b, h)
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping: kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] vector
        q_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Compute lse components
        max_s = -float("inf")
        sum_exp = 0.0  # scalar float32

        # Pass 1: compute max and sum(exp(s - max)) across tokens
        start = tl.load(kv_indptr_ptr + b)     # int32
        end = tl.load(kv_indptr_ptr + b + 1)   # int32
        num_tokens_actual = end - start        # int32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows for this kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32
        # Store lse for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar
            out_vec += attn * v_vec     # masked: if mask_i is False, v_vec is zero

        # Store output[b, h, :]
        out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off + tl.arange(0, HEAD_DIM), out_vec)

    def _run_triton(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity, cast to float32 for compute
        q = q.contiguous().to(torch.float32)
        k_cache = k_cache.contiguous().to(torch.float32)
        v_cache = v_cache.contiguous().to(torch.float32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, num_k_cache_heads, num_kv_heads, d = k_cache.shape
        assert num_k_cache_heads == 1 and num_kv_heads == 8 and d == head_dim, "Cache shape must be [num_pages, 1, 8, 128]"

        # Output buffers (float32 for compute, cast later)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        # Choose a conservative bound for tokens; mask out beyond actual
        # Use 8192 to cover typical workloads
        num_toks = 8192
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse, sm_scale,
            num_qo_heads, num_kv_heads=8, head_dim=128, num_toks=num_toks, half_ln2_inv=half_ln2_inv,
            num_warps=4, num_stages=2
        )

        return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Use Triton when available and tensors are on CUDA
        if TRITON_AVAILABLE and q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda:
            output_f32, lse = _run_triton(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)
            # Match original output dtypes: output in bfloat16, lse in float32
            output = output_f32.to(torch.bfloat16)
            return output, lse
        else:
            # Fallback: original PyTorch logic (rare in evaluation)
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            len_indptr = kv_indptr.shape[0]
            device = q.device

            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )

            gqa_ratio = num_qo_heads // num_kv_heads

            k_cache_flat = k_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]
            v_cache_flat = v_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                if start >= end:
                    output[b].zero_()
                    continue

                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    output[b].zero_()
                    continue

                k_batch = k_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
                v_batch = v_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
                q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [head_dim]
                    k_head = k_batch[:, kv_head]  # [num_tokens, head_dim]
                    v_head = v_batch[:, kv_head]  # [num_tokens, head_dim]

                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale

                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_head)  # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse


def run(*args):
    return ModelNew()(*args)
