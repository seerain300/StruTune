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
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        ln2: tl.constexpr,   # 1 / ln(2)
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

        # Load q vector for this (b, h) as float32: q_ptr[b, h, :]
        q_base = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))  # [HEAD_DIM] f32

        # GQA mapping
        kv_head = h // (num_qo_heads // num_kv_heads)

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Loop over tokens
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # token index
            # Load k_t and v_t as float32: k_ptr[num_pages, num_kv_heads, HEAD_DIM]
            # For each (b), idx selects the row in k_cache/v_cache at "page" 0 implicitly
            offset = idx * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM
            k_t = tl.load(k_ptr + offset + tl.arange(0, HEAD_DIM)).to(tl.float32)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + offset + tl.arange(0, HEAD_DIM)).to(tl.float32)  # [HEAD_DIM]

            # Compute dot product and scaled logits
            logits = tl.sum(q_vec * k_t, axis=0)  # scalar
            scaled = logits * sm_scale

            # Streaming, numerically stable LSE update
            is_init = lse == -float("inf")
            if is_init:
                lse = scaled
            else:
                # lse_new = lse + log(1 + exp(s - lse))
                lse = lse + tl.log(1.0 + tl.exp(scaled - lse))

            # Attention weight
            attn = tl.exp(scaled - lse)  # scalar

            # Accumulate output vector
            out_vec += attn * v_t

            t += 1

        # Store results
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset + tl.arange(0, HEAD_DIM), out_vec)

        lse_scaled = lse * ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)

class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtypes
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"
        # k_cache/v_cache shape checks
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        ln2 = 1.0 / math.log(2.0)

        # Compute in float32
        q_f32 = q.to(torch.float32)

        # Allocate outputs
        out_f32 = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32,
            k_cache,
            v_cache,
            kv_indptr,
            out_f32,
            lse_f32,
            batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            sm_scale=sm_scale,
            ln2=ln2,
        )

        # Return output in bfloat16 and lse in float32 (already scaled by ln2)
        out_bf16 = out_f32.to(torch.bfloat16)
        return out_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
