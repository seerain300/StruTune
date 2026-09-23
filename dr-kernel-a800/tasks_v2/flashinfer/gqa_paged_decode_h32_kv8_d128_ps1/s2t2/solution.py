import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS tokens with masks, computing:
# - lse = logsumexp(s) / ln(2), where s = q[h] @ k_i * sm_scale
# - output vector out = sum_i exp(s_i - lse) * v_i
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,           # *float32, shape [B, 32, 128]
        k_ptr,           # *float32, shape [num_pages, 8, 128]
        v_ptr,           # *float32, shape [num_pages, 8, 128]
        kv_indptr_ptr,   # *int32,   shape [B+1]
        kv_indices_ptr,  # *int32,   shape [num_tokens]
        out_ptr,         # *bfloat16, shape [B, 32, 128]
        lse_ptr,         # *float32, shape [B, 32]
        sm_scale,        # float32 scalar
        half_ln2_inv,    # float32 scalar = 1 / ln(2) = 1.4426950408889634
        B: tl.constexpr, NUM_QO_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr, NUM_TOKS: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        b = pid // NUM_QO_HEADS
        h = pid % NUM_QO_HEADS

        # Base offsets
        q_off = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM

        # Load q vector for this (b, h)
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Accumulators for lse (logsumexp(s) / ln(2))
        max_s = tl.full([], -float("inf"), tl.float32)
        sum_exp = tl.full([], 0.0, tl.float32)

        # Compute start/end from kv_indptr
        start = tl.load(kv_indptr_ptr + b)       # int32
        end = tl.load(kv_indptr_ptr + b + 1)     # int32
        num_tokens_actual = end - start          # int32

        # Pass 1: accumulate max and sum(exp(s - max)) across tokens
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # GQA mapping: kv_head = h // (32 // 8) = h // 4
            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio  # 0..7

            # Offsets for k and v rows
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

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

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32

            # Accumulate output vector only if valid
            if mask_i:
                out_vec += attn * v_vec

        # Store output as bfloat16 and lse as float32
        out_base = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec.to(tl.bfloat16))
        tl.store(lse_ptr + (b * NUM_QO_HEADS + h), lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback to PyTorch if Triton is unavailable
        if not TRITON_AVAILABLE:
            B, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, \
                "This implementation expects num_qo_heads=32, num_kv_heads=8, head_dim=128"
            output = torch.zeros((B, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((B, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

            for b in range(B):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens == 0:
                    output[b].zero_()
                    lse[b, :] = -float("inf")
                    continue

                token_indices = kv_indices[start:end].to(torch.long)
                k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
                v_cache_flat = v_cache.squeeze(1).to(torch.float32)
                q_batch = q[b].to(torch.float32)  # [32, 128]

                for h in range(num_qo_heads):
                    kv_head = h // (num_qo_heads // num_kv_heads)
                    q_head = q_batch[h]                        # [128]
                    k_head = k_cache_flat[token_indices]      # [num_tokens, 8, 128]
                    v_head = v_cache_flat[token_indices]      # [num_tokens, 8, 128]

                    logits = torch.matmul(q_head, k_head.T)   # [num_tokens]
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

        # Prepare inputs
        q32 = q.contiguous().to(torch.float32)                 # [B, 32, 128]
        k_flat32 = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        v_flat32 = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)

        # Output and lse
        out = torch.empty((B, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Handle zero-token batches on host to avoid kernel branching
        zero_batches = []
        for b in range(B):
            start = int(kv_indptr_i32[b].item())
            end = int(kv_indptr_i32[b + 1].item())
            num_tokens = end - start
            if num_tokens == 0:
                zero_batches.append(b)

        if zero_batches:
            for b in zero_batches:
                out[b].zero_()
                lse[b, :] = -float("inf")

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
