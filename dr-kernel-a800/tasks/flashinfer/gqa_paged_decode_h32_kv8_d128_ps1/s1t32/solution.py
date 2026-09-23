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
    def _gqa_attention_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # natural log of 2
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float('inf')

        # Iterate over tokens
        for t in range(0, 1024):  # MAX_TOKENS; kernel breaks when t >= num_tokens
            if t >= num_tokens:
                break
            idx = kv_start + t  # i32

            # Gather k_t and v_t as float32
            kv_head = h // gqa_ratio
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_t = tl.load(k_ptr + k_offset)  # [HEAD_DIM] (bf16/f16), cast to f32
            v_t = tl.load(v_ptr + v_offset)  # [HEAD_DIM] (bf16/f16), cast to f32
            k_t = k_t.to(tl.float32)
            v_t = v_t.to(tl.float32)

            # Dot product q_vec · k_t
            logits = 0.0
            for d in range(HEAD_DIM):
                logits += q_vec[d] * k_t[d]

            # Scale logits
            scaled = logits * sm_scale

            # Update LSE stably
            # if lse == -inf: lse = scaled
            # else: lse = max(lse, scaled) + log(1 + exp(scaled - lse))
            is_neg_inf = lse == -float('inf')
            if is_neg_inf:
                lse = scaled
            else:
                m = tl.maximum(lse, scaled)
                z = tl.exp(scaled - lse)
                lse = m + tl.log(1.0 + tl.exp(lse - m + tl.log(1.0 + z)))

            # Compute attention and accumulate
            attn = tl.exp(scaled - lse)  # f32
            out_vec += attn * v_t

        # Store output and lse
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_scaled = lse * (1.0 / ln2)  # divide by ln(2)
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA for Triton
        device = q.device
        # Shapes and constants
        B, num_qo_heads, HEAD_DIM = q.shape
        assert num_qo_heads == 32, "Expected num_qo_heads == 32"
        num_kv_heads = k_cache.shape[2]
        assert num_kv_heads == 8, "Expected num_kv_heads == 8"
        assert HEAD_DIM == 128, "Expected head_dim == 128"
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = 1.0 / math.log(2.0)

        # Compute in float32; output as bfloat16, lse as float32 divided by ln(2)
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr,
            output, lse,
            B=B, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM, sm_scale=float(sm_scale), gqa_ratio=gqa_ratio,
            ln2=ln2,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 as original model returns
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
