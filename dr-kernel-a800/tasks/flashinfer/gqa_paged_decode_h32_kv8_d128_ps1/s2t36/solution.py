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
        q_ptr,            # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_KV_INDICES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_KV_INDICES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,
        SM_SCALE: tl.float32,
        HALF_LN2_INV: tl.float32,  # 1 / ln(2) = 1.442695...
    ):
        # program ids: one program per (b, h)
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping: kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] vector: q[b, h, :] -> [HEAD_DIM]
        q_base = q_ptr + b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_base + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # First pass: compute max_s and sum_exp = sum(exp(s - max_s)) across tokens
        start = tl.load(kv_indptr_ptr + b)     # int32
        end = tl.load(kv_indptr_ptr + b + 1)   # int32
        num_tokens_actual = end - start        # int32

        max_s = -float("inf")
        sum_exp = 0.0  # scalar float32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            # Load token index
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows for this kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * SM_SCALE

            # Update max and sum-exp only if valid
            if mask_i:
                # Re-express sum_exp update in a branch-free style:
                # if new max, recompute sum_exp = sum(exp(s_i - new_max))
                # else, sum_exp_new = sum_exp * exp(max_s - s) + 1
                # We can implement this update with tl.where for clarity.
                new_max = tl.maximum(max_s, s)
                # Compute old contribution if max didn't change, else reset
                old_sum = sum_exp * tl.exp(max_s - s) + 1.0 if (max_s >= s) else 0.0
                sum_exp = tl.where(new_max == max_s, sum_exp * tl.exp(max_s - s) + 1.0, sum_exp)
                max_s = new_max

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * HALF_LN2_INV  # float32
        # Store lse for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Second pass: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product and scaled logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * SM_SCALE

            # attn_i = exp(s - lse) when valid, else 0
            attn_i = tl.exp(s - lse_val) if mask_i else 0.0

            # Accumulate output
            out_vec += attn_i * v_vec

        # Store output vector for this (b, h)
        tl.store(out_ptr + b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback if Triton not available
        if not TRITON_AVAILABLE:
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

            k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [P, 8, 128]
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [P, 8, 128]

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]

                q_batch = q[b].to(torch.float32)  # [32, 128]

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [128]
                    k_head = k_cache_flat[token_indices, kv_head]  # [num_tokens, 128]
                    v_head = v_cache_flat[token_indices, kv_head]  # [num_tokens, 128]

                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale

                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)
                    out_head = torch.matmul(attn, v_head)
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path
        device = q.device

        # Ensure tensors are contiguous and float32 for compute
        q32 = q.contiguous().to(torch.float32)               # [B, 32, 128]
        k32 = k_cache.contiguous().to(torch.float32).squeeze(1)         # [P, 8, 128]
        v32 = v_cache.contiguous().to(torch.float32).squeeze(1)         # [P, 8, 128]
        kv_indptr32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices32 = kv_indices.contiguous().to(torch.int32)

        batch_size = q32.shape[0]
        num_qo_heads = q32.shape[1]
        head_dim = q32.shape[2]
        num_kv_heads = k32.shape[1]

        # Output buffers (float32 for compute, then cast to bfloat16)
        out32 = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse32 = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # Conservative token bound for masks
        NUM_TOKS = 8192
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k32, v32,
            kv_indptr32, kv_indices32,
            out32, lse32,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            SM_SCALE=float(sm_scale),
            HALF_LN2_INV=half_ln2_inv,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original, lse stays float32
        output = out32.to(torch.bfloat16)
        return output, lse32


def run(*args):
    return ModelNew()(*args)
