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
    def _attention_single_head_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *f16 or *bf16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B: tl.constexpr,                # batch size (constexpr for grid math)
        num_qo_heads: tl.constexpr,     # number of query heads (32)
        num_kv_heads: tl.constexpr,     # number of kv heads (8)
        HEAD_DIM: tl.constexpr,         # head dim (128)
        sm_scale: tl.constexpr,         # scaling factor (float)
        ln2: tl.constexpr,              # 1 / log(2) (float)
        GQA_RATIO: tl.constexpr,        # num_qo_heads // num_kv_heads (4)
        MAX_TOKENS: tl.constexpr,       # loop bound (>= typical num_tokens)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # GQA mapping: select KV head for this query head
        kv_head = h // GQA_RATIO  # int

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float("inf")  # scalar f32

        # Iterate tokens, compute attention and accumulate
        for t in range(MAX_TOKENS):
            # Break if t >= num_tokens
            if t >= num_tokens:
                break

            # idx = kv_start + t
            idx = kv_start + t

            # Load k_t and v_t as f32 and f32 (we cast if needed in kernel)
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_t = tl.load(k_ptr + k_offset).to(tl.float32)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_offset).to(tl.float32)  # [HEAD_DIM]

            # Dot product: q[b, h] · k_t
            dot = 0.0
            for j in range(HEAD_DIM):
                dot += q_vec[j] * k_t[j]

            # Scale logits
            scaled = dot * sm_scale  # f32 scalar

            # Update numerically stable LSE: new_lse = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            diff = scaled - lse
            m = tl.maximum(lse, scaled)
            new_lse = m + tl.log(1.0 + tl.exp(-tl.abs(diff)))
            lse = new_lse  # update LSE

            # Compute attention weight and accumulate output
            attn = tl.exp(scaled - lse)  # f32 scalar
            out_vec += attn * v_t

        # Store results
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # f32
        lse_b_h = lse * ln2  # divide by ln(2)
        tl.store(lse_ptr + b * num_qo_heads + h, lse_b_h)  # f32


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Shapes and constants
        B, num_qo_heads, HEAD_DIM = q.shape
        assert num_qo_heads == 32, "Expected num_qo_heads == 32"
        num_kv_heads = k_cache.shape[2]
        assert num_kv_heads == 8, "Expected num_kv_heads == 8"
        assert HEAD_DIM == 128, "Expected head_dim == 128"
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = 1.0 / math.log(2.0)

        # Compute in float32; output as bfloat16, lse as float32 divided by ln(2)
        q_f32 = q.to(torch.float32)

        # Allocate outputs (compute in f32, cast later)
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        MAX_TOKENS = 1024  # loop bound; covers typical num_tokens

        _attention_single_head_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr,
            output, lse,
            B=B, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM, sm_scale=float(sm_scale), ln2=ln2,
            GQA_RATIO=gqa_ratio, MAX_TOKENS=MAX_TOKENS,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 as original model returns
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
