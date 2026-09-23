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
        q_ptr,                 # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,                 # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,                 # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,         # *int32, shape [BATCH_SIZE + 1]
        kv_indices_ptr,        # *int32, shape [NUM_KV_INDICES]
        out_ptr,               # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,               # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,              # float32 scalar (e.g., 1/sqrt(128))
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,          # upper bound for tokens per batch (e.g., 8192)
        HALF_LN2_INV: tl.constexpr,      # 1 / ln(2) = 1.4426950408889634
    ):
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Determine token range for this batch
        start = tl.load(kv_indptr_ptr + b)             # int32
        end = tl.load(kv_indptr_ptr + b + 1)          # int32
        num_tokens_actual = end - start               # int32 (can be 0)

        # Load q[h] vector
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Pass 1: compute max and sum(exp(s - max)) across tokens
        max_s = tl.full((), -1.0e20, tl.float32)
        sum_exp = tl.zeros((), tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            # Load index for this token
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32
            # Offsets for k and v rows: idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            # Compute logits = q[h] · k_i
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1 / ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * HALF_LN2_INV
        # Store lse[b, h]
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

            # logits and scaled
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)              # scalar float32

            # accumulate
            out_vec += attn * v_vec

        # Store output[b, h, :]
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton unavailable or not CUDA, fall back to PyTorch (not ideal, but safe)
        if not TRITON_AVAILABLE or not q.is_cuda:
            # Fallback: keep behavior identical to original
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            device = q.device

            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )

            gqa_ratio = num_qo_heads // num_kv_heads
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]

                q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [head_dim]
                    k_heads = k_cache_flat[token_indices, kv_head]  # [num_tokens, head_dim]
                    v_heads = v_cache_flat[token_indices, kv_head]  # [num_tokens, head_dim]

                    logits = torch.matmul(q_head, k_heads.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale

                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_heads)  # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path: ensure CUDA, contiguous, float32
        device = q.device
        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # Ensure inputs are contiguous and float32 for compute
        q32 = q.contiguous().to(torch.float32)
        k32 = k_cache.contiguous().to(torch.float32)
        v32 = v_cache.contiguous().to(torch.float32)
        kv_indptr32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices32 = kv_indices.contiguous().to(torch.int32)

        # Output buffers (float32 for compute), cast later to bfloat16
        out = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        # Choose an upper bound for tokens; adjust as needed. 8192 is safe for the given workloads.
        NUM_TOKS = 8192
        HALF_LN2_INV = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k32, v32,
            kv_indptr32, kv_indices32,
            out, lse,
            sm_scale,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            HALF_LN2_INV=HALF_LN2_INV,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original
        output_bf16 = out.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
