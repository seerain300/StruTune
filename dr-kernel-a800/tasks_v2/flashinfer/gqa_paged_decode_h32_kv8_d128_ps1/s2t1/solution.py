import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS tokens with masks, accumulating:
# - lse = logsumexp(logits_scaled) / ln(2) where logits_scaled = q @ k_i * sm_scale
# - output vector out = sum_i attn_i * v_i, with attn_i = exp(logits_scaled_i - lse)
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,           # *float32, [B, 32, 128]
        k_ptr,           # *float32, [num_pages, 8, 128]
        v_ptr,           # *float32, [num_pages, 8, 128]
        kv_indptr_ptr,   # *int32,   [B+1]
        kv_indices_ptr,  # *int32,   [num_tokens]
        out_ptr,         # *bfloat16, [B, 32, 128]
        lse_ptr,         # *float32, [B, 32]
        sm_scale,        # float32 scalar
        half_ln2_inv,    # float32 scalar = 1 / ln(2) = 1.4426950408889634
        B: tl.constexpr, NUM_QO_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr, NUM_TOKS: tl.constexpr  # upper bound for tokens
    ):
        pid = tl.program_id(axis=0)
        b = pid // NUM_QO_HEADS
        h = pid % NUM_QO_HEADS

        # Load q vector for this (b, h)
        q_off = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Indptr for this batch
        start = tl.load(kv_indptr_ptr + b)      # int32
        end = tl.load(kv_indptr_ptr + b + 1)    # int32
        num_tokens_actual = end - start         # int32 (scalar)

        # Accumulators for lse
        max_s = tl.full([], -float("inf"), tl.float32)
        sum_exp = tl.full([], 0.0, tl.float32)

        # Pass 1: compute max and sumexp of s = q @ k_i * sm_scale
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio  # 0..7

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product and scale
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sumexp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = logsumexp(s) / ln(2) = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Pass 2: recompute s and accumulate output
        out = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)

            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # softmax over tokens
            out = out + attn * v_vec

        # Store output (bfloat16) and lse (float32)
        out_base = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_base, out.to(tl.bfloat16))
        tl.store(lse_ptr + (b * NUM_QO_HEADS + h), lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback to PyTorch if Triton is not available or not on CUDA
        if (not TRITON_AVAILABLE) or (q.device.type != "cuda"):
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, \
                "This implementation expects num_qo_heads=32, num_kv_heads=8, head_dim=128"
            device = q.device

            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )

            k_cache_flat = k_cache.squeeze(1).to(torch.float32)
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)

            gqa_ratio = num_qo_heads // num_kv_heads

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens == 0:
                    output[b].zero_()
                    lse[b, :] = -float("inf")
                    continue

                token_indices = kv_indices[start:end].to(torch.long)
                k_batch = k_cache_flat[token_indices]  # [num_tokens, 8, 128]
                v_batch = v_cache_flat[token_indices]  # [num_tokens, 8, 128]
                q_batch = q[b].to(torch.float32)      # [32, 128]

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]              # [128]
                    k_head = k_batch[:, kv_head]     # [num_tokens, 128]
                    v_head = v_batch[:, kv_head]     # [num_tokens, 128]

                    logits = torch.matmul(q_head, k_head.T)     # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse0 = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_head)       # [128]
                    output[b, h] = out_head.to(torch.bfloat16)
                    lse[b, h] = lse0.item()

            return output, lse

        # Triton path
        device = q.device
        B, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, \
            "This Triton kernel expects num_qo_heads=32, num_kv_heads=8, head_dim=128"
        assert kv_indptr.shape[0] == B + 1

        # Prepare tensors
        q32 = q.contiguous().to(torch.float32)          # [B, 32, 128]
        k_flat32 = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        v_flat32 = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)

        # Output and lse
        out = torch.zeros((B, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((B, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # For each batch, if num_tokens == 0, skip kernel and write zeros
        for b in range(B):
            start = int(kv_indptr_i32[b].item())
            end = int(kv_indptr_i32[b + 1].item())
            num_tokens = end - start
            if num_tokens == 0:
                out[b].zero_()
                lse[b, :] = -float("inf")
                continue

        # If all batches have zero tokens, we can return early
        if all((int(kv_indptr_i32[b + 1].item()) - int(kv_indptr_i32[b].item())) == 0 for b in range(B)):
            return out, lse

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        NUM_TOKS = 8192  # conservative upper bound; masked in the kernel
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k_flat32, v_flat32, kv_indptr_i32, kv_indices_i32,
            out, lse,
            sm_scale, half_ln2_inv,
            B=B, NUM_QO_HEADS=num_qo_heads, NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim, NUM_TOKS=NUM_TOKS,
        )

        return out, lse


def run(*args):
    return ModelNew()(*args)
