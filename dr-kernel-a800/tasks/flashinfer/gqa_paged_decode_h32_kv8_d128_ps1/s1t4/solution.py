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
        lse = -float("inf")

        # Loop over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index into k/v cache for this batch

            # GQA mapping: KV head corresponding to query head h
            kv_head = h // gqa_ratio  # 32 // 8 = 4

            # Load k_t and v_t as original dtype, then cast k_t to f32 for compute
            # k_ptr/v_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM]
            # For a given idx and kv_head, linear offset is: idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_base = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_base = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_t = tl.load(k_ptr + k_base)  # [HEAD_DIM] (bf16/f16)
            v_t = tl.load(v_ptr + v_base)  # [HEAD_DIM] (bf16/f16)

            # Compute dot product q_vec · k_t
            k_t_f32 = k_t.to(tl.float32)
            qk = 0.0
            for d in range(HEAD_DIM):
                qd = q_vec[d]
                kd = k_t_f32[d]
                qk += qd * kd

            # Scale logits
            scaled = qk * sm_scale

            # Streaming LSE update (stable)
            if lse == -float("inf"):
                lse = scaled
            else:
                m = tl.maximum(lse, scaled)
                diff = m - lse
                new = m + tl.log(1.0 + tl.exp(-diff))
                lse = new

            # Compute attention
            attn = tl.exp(scaled - lse)

            # Accumulate output
            out_vec += attn * v_t.to(tl.float32)

        # Store output and lse
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)
        tl.store(lse_ptr + b * num_qo_heads + h, lse / ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and constants
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = math.log(2.0)

        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()          # compute q in f32
        k_cache_c = k_cache.contiguous()
        v_cache_c = v_cache.contiguous()
        kv_indptr_c = kv_indptr.contiguous()

        # Allocate outputs (compute in f32, store as bf16 and lse as f32)
        output = torch.empty(
            (batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device
        )
        lse = torch.empty(
            (batch_size, num_qo_heads), dtype=torch.float32, device=q.device
        )

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache_c, v_cache_c, kv_indptr_c,
            output, lse,
            batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            sm_scale=float(sm_scale),
            gqa_ratio=gqa_ratio,
            ln2=ln2,
        )

        # Return output as bfloat16 (matching original) and lse (float32 divided by ln(2))
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
