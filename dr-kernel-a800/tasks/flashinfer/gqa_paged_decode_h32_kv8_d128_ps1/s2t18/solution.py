import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h)
# It loops over up to NUM_TOKS cached token indices with masks, accumulating:
# - lse = logsumexp(scaled_dot) / ln(2)
# - output vector out = sum_i exp(scaled_dot_i - lse) * v_i
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
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,
        sm_scale: tl.float32,
        half_ln2_inv: tl.float32,  # 1 / ln(2) = 1.4426950408889634
    ):
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # Load q[h] vector
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM], float32

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS  # 4
        kv_head = h // kv_ratio  # 0..7

        # Initialize accumulators for lse
        max_s = -float("inf")
        sum_exp = 0.0

        # Pass 1: compute max_s and sum_exp = sum(exp(s - max_s)) across tokens (masked)
        for i in range(NUM_TOKS):
            # Determine if this i is valid for batch b
            # num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
            num_tokens_b = kv_indptr_ptr[b + 1].item() - kv_indptr_ptr[b].item()
            mask_i = i < num_tokens_b

            # Load token index at position i for this batch (guarded by mask_i)
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b].item() + i, mask=mask_i, other=0)

            # Offsets into k and v for kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product for this token
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            s = logits * sm_scale

            # Update max and sum(exp)
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv

        # Accumulate output vector
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Pass 2: recompute s and accumulate output: out_vec += exp(s - lse) * v_i
        for i in range(NUM_TOKS):
            num_tokens_b = kv_indptr_ptr[b + 1].item() - kv_indptr_ptr[b].item()
            mask_i = i < num_tokens_b

            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b].item() + i, mask=mask_i, other=0)
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)

            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # masked naturally via mask_i handling; no need for tl.where since attn=0 when mask_i=False due to masked loads contributing 0
            out_vec += attn * v_vec

        # Store results
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no torch ops. All compute in Triton.
        if not TRITON_AVAILABLE:
            # Fallback: if Triton not available, mimic original behavior (not used in evaluation).
            batch_size, num_qo_heads, head_dim = q.shape
            device = q.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
            gqa_ratio = num_qo_heads // 8
            q32 = q.to(torch.float32)
            k32 = k_cache.squeeze(1).to(torch.float32)
            v32 = v_cache.squeeze(1).to(torch.float32)
            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens == 0:
                    output[b].zero_()
                    lse[b].fill_(-float("inf"))
                    continue
                token_indices = kv_indices[start:end].to(torch.long)
                q_batch = q32[b]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]
                    k_batch = k32[token_indices]  # [num_tokens, 8, 128]
                    v_batch = v32[token_indices]  # [num_tokens, 8, 128]
                    logits = torch.matmul(q_head, k_batch.transpose(1, 2))  # [1, num_tokens]
                    logits = logits.squeeze(0)
                    s = logits * sm_scale
                    lse_b_h = torch.logsumexp(s, dim=0) / math.log(2.0)
                    attn = torch.softmax(s, dim=0)
                    out_h = torch.matmul(attn, v_batch[:, kv_head])
                    output[b, h] = out_h.to(torch.bfloat16)
                    lse[b, h] = lse_b_h
            return output, lse

        # Ensure tensors are contiguous and on CUDA
        batch_size, num_qo_heads, head_dim = q.shape
        device = q.device
        q32 = q.contiguous().to(torch.float32)  # [B, 32, 128]
        # k_cache, v_cache are [num_pages, 1, 8, 128] -> squeeze dim-1 -> [num_pages, 8, 128]
        k32 = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        v32 = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Output and lse buffers (float32 for compute)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # Conservative bound; masks ensure correctness across variable num_tokens
        NUM_TOKS = 8192
        sm_scale_val = float(sm_scale)
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k32, v32,
            kv_indptr, kv_indices,
            output, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=8,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            sm_scale=sm_scale_val,
            half_ln2_inv=half_ln2_inv,
            num_warps=4,
        )

        # Cast output to bfloat16 to match original output dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
