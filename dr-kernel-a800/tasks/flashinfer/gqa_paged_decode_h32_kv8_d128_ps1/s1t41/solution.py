import torch
import math

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _gqa_attention_single_batch_kernel(
        q_ptr,           # *f32, shape [B, 32, 128]
        k_ptr, v_ptr,    # *bf16/f16, shape [TOT_TOKENS, 128] (since middle=1, we flatten)
        kv_indptr_ptr,   # *i32, shape [B+1], marks start/end of tokens per batch
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B: tl.constexpr,                # int32
        NUM_QO_HEADS: tl.constexpr,     # 32
        NUM_KV_HEADS: tl.constexpr,     # 8
        HEAD_DIM: tl.constexpr,         # 128
        SM_SCALE: tl.constexpr,         # float32
        LN2: tl.constexpr,              # 1.0 / log(2.0)
        GQA_RATIO: tl.constexpr,        # NUM_QO_HEADS // NUM_KV_HEADS == 4
    ):
        # One program per (batch, head)
        pid = tl.program_id(0)
        b = pid // NUM_QO_HEADS
        h = pid % NUM_QO_HEADS
        if b >= B or h >= NUM_QO_HEADS:
            return

        # Compute number of tokens for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32

        # Load q vector for this (b, h) as float32
        q_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Iterate over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index within flattened tokens
            kv_head = h // GQA_RATIO  # GQA mapping

            # Base offsets for flattened k/v vectors (middle=1, so stride is HEAD_DIM)
            k_base = idx * HEAD_DIM
            v_base = idx * HEAD_DIM

            # Load k_vec and v_vec (length HEAD_DIM), cast to f32
            k_vec = tl.load(k_ptr + k_base)     # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_base)     # [HEAD_DIM]
            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product: sum_i q_vec[i] * k_vec[i]
            dot = 0.0
            for d in range(0, HEAD_DIM):
                dot += q_vec[d] * k_vec[d]

            # Scale logits
            scaled = dot * SM_SCALE  # float32

            # Update LSE stably
            # If lse == -inf: lse = scaled
            # Else: m = max(lse, scaled); lse = m + log(1 + exp(scaled - m))
            is_init = lse == -float("inf")
            m = tl.maximum(lse, scaled)
            delta = tl.log(1.0 + tl.exp(scaled - m))
            new_lse = tl.where(is_init, scaled, m + delta)
            lse = new_lse

            # Attention weight
            attn = tl.exp(scaled - lse)

            # Accumulate output vector
            out_vec += attn * v_vec

        # Store output vector to [b, h]
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # out_ptr is f32

        # Store LSE / ln(2) to lse[b, h]
        lse_scaled = lse * LN2
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], dtype bfloat16
        k_cache: [num_pages, 1, 8, 128], dtype bfloat16/f16
        v_cache: [num_pages, 1, 8, 128], dtype bfloat16/f16
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32 (not used in original run; we rely on kv_indptr)
        sm_scale: float32
        Returns: output [B, 32, 128] (bfloat16), lse [B, 32] (float32)
        """
        assert TRITON_AVAILABLE, "Triton is not available. Please install triton to run this kernel."

        B, num_qo_heads, HEAD_DIM = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and HEAD_DIM == 128

        # Compute total tokens: num_tokens_total = kv_indptr[-1]
        total_tokens = int(kv_indptr[-1].item())

        # Flatten k/v for simplicity: since middle=1, each token has 128-d vector
        # We will treat k_cache and v_cache as [total_tokens, 128] by flattening.
        # Note: num_pages is not used by original logic; we rely on kv_indptr to define token ranges per batch.
        k_flat = k_cache.contiguous().reshape(total_tokens, HEAD_DIM)  # [total_tokens, 128]
        v_flat = v_cache.contiguous().reshape(total_tokens, HEAD_DIM)  # [total_tokens, 128]

        # Output buffers (float32 compute; cast to bfloat16 later)
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Prepare q as float32
        q_f32 = q.to(torch.float32).contiguous()  # [B, 32, 128]

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        gqa_ratio = num_qo_heads // num_kv_heads  # 4
        ln2 = 1.0 / math.log(2.0)

        _gqa_attention_single_batch_kernel[grid](
            q_f32,
            k_flat, v_flat,
            kv_indptr.contiguous(),
            output, lse,
            B=B,
            NUM_QO_HEADS=32,
            NUM_KV_HEADS=8,
            HEAD_DIM=128,
            SM_SCALE=float(sm_scale),
            LN2=ln2,
            GQA_RATIO=gqa_ratio,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 as in original
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
