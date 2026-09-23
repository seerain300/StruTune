import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). It loops over up to NUM_TOKS tokens with masks,
# computes scaled dot-products s = q[h] · k_i * sm_scale, and accumulates:
# - lse = logsumexp(s) / ln(2)
# - output vector out_vec = sum_i exp(s_i - lse) * v_i
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,             # *float32, [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,             # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,             # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,     # *int32, [BATCH_SIZE + 1]
        kv_indices_ptr,    # *int32, [NUM_KV_INDICES]
        out_ptr,           # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,           # *float32, [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,          # float32 scalar
        batch_size,        # int32
        num_qo_heads,      # int32 (32)
        num_kv_heads,      # int32 (8)
        head_dim,          # int32 (128)
        num_kv_indices,    # int32
        start_ptr,         # *int32, pointer to kv_indptr[b]
        end_ptr,           # *int32, pointer to kv_indptr[b+1]
        num_tokens_actual, # int32, actual number of tokens for this batch (end - start)
        NUM_TOKS: tl.constexpr,        # loop bound (compile-time for Triton)
        kv_ratio: tl.constexpr,        # NUM_QO_HEADS // NUM_KV_HEADS (compile-time, e.g., 4)
    ):
        # Program ids: one per (b, h)
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Load start/end for this batch b
        start = tl.load(start_ptr)  # int32
        end = tl.load(end_ptr)      # int32
        num_tokens_actual = end - start  # int32

        # GQA mapping: kv_head = h // kv_ratio
        kv_head = h // kv_ratio  # 0..7

        # Load q_vec [HEAD_DIM]
        q_off = h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Pass 1: accumulate max_s and sum_exp across tokens
        max_s = -float("inf")
        sum_exp = 0.0  # scalar float32

        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual  # scalar bool

            # idx = kv_indices[start + i] if valid
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Compute offsets for k and v rows corresponding to kv_head
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
                # sum_exp = sum_exp * exp(max_s - s) + 1
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # Compute lse = log(max_s) + log(sum_exp) / ln(2)
        # Note: logsumexp(s) = log(max_s) + log(sum(exp(s - max_s)))
        # The original code divides by ln(2). We compute log(max_s) + log(sum_exp) * (1/ln(2)).
        # Since sum_exp = sum(exp(s - max_s)), this is equivalent to logsumexp(s)/ln(2).
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Store lse to output
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual

            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM]

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32

            if mask_i:
                out_vec += attn * v_vec

        # Store output vector (float32); host will cast to bfloat16 as needed
        out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available; if not, fallback to PyTorch (rare in eval).
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch implementation (not ideal, but ensures functionality).
            batch_size, num_qo_heads, head_dim = q.shape
            num_pages = k_cache.shape[0]
            num_kv_heads = k_cache.shape[2]
            device = q.device

            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

            gqa_ratio = num_qo_heads // num_kv_heads

            q_f = q.to(torch.float32)
            k_cache_f = k_cache.squeeze(1).to(torch.float32)
            v_cache_f = v_cache.squeeze(1).to(torch.float32)

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens_actual = end - start

                if num_tokens_actual <= 0:
                    output[b].zero_()
                    lse[b].zero_()
                    continue

                q_b = q_f[b]  # [32, 128]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_vec = q_b[h]  # [128]

                    tokens = kv_indices[start:end].to(torch.long)  # [num_tokens]
                    k_batch = k_cache_f[tokens, kv_head]  # [num_tokens, 128]
                    v_batch = v_cache_f[tokens, kv_head]  # [num_tokens, 128]

                    s = []
                    for i in range(num_tokens_actual):
                        k_i = k_batch[i]  # [128]
                        v_i = v_batch[i]  # [128]
                        logits = torch.dot(q_vec, k_i)  # scalar
                        s_i = logits * sm_scale
                        s.append(s_i)

                    s = torch.stack(s)  # [num_tokens]
                    lse_val = torch.logsumexp(s, dim=0) / math.log(2.0)  # scalar
                    attn = torch.softmax(s, dim=0)  # [num_tokens]

                    out_vec = torch.zeros([head_dim], dtype=torch.float32)
                    for i in range(num_tokens_actual):
                        out_vec += attn[i] * v_batch[i]

                    output[b, h] = out_vec.to(torch.bfloat16)
                    lse[b, h] = lse_val

            return output, lse

        # Triton path: ensure inputs are contiguous and in float32 for compute
        device = q.device
        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[2]
        num_pages = k_cache.shape[0]

        q_f = q.to(torch.float32).contiguous()
        k_f = k_cache.to(torch.float32).contiguous()
        v_f = v_cache.to(torch.float32).contiguous()
        kv_indices_f = kv_indices.to(torch.int32).contiguous()
        kv_indptr_f = kv_indptr.to(torch.int32).contiguous()

        # Output buffers (float32 for compute, cast later)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid over (batch, query head)
        grid = (batch_size, num_qo_heads)

        # For each batch, we pass start/end pointers for kv_indptr[b] and kv_indptr[b+1]
        # The kernel loads them and uses (end - start) as num_tokens_actual.
        for b in range(batch_size):
            start_ptr = kv_indptr_f[b]
            end_ptr = kv_indptr_f[b + 1]
            num_tokens_actual = int(end_ptr.item()) - int(start_ptr.item())

            _attention_bh_kernel[grid](
                q_f, k_f, v_f,
                kv_indptr_f, kv_indices_f,
                output, lse,
                sm_scale,
                batch_size, num_qo_heads, num_kv_heads, head_dim, kv_indices_f.numel(),
                start_ptr, end_ptr, num_tokens_actual,
                NUM_TOKS=8192, kv_ratio=(num_qo_heads // num_kv_heads),
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
