import torch
import math

# Try to import Triton; define kernels for Triton path
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per (b, h) computes attention and stores output vector and LSE
if TRITON_AVAILABLE:
    @triton.jit
    def _gqa_attention_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM] (we'll cast to bfloat16 in Python)
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B: tl.constexpr,
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # 1 / ln(2)
        MAX_TOKENS: tl.constexpr,  # loop bound for token iteration
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
        lse = tl.full((), -float("inf"), tl.float32)

        # Iterate tokens
        for t in range(0, MAX_TOKENS):
            if t >= num_tokens:
                break
            idx = kv_start + t  # token index in cache
            kv_head = h // gqa_ratio  # GQA mapping

            # Compute offsets for k/v at this idx and kv_head
            # k_ptr/v_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM]
            # With contiguous, offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            base = num_kv_heads * HEAD_DIM
            k_offset = idx * base + kv_head * HEAD_DIM
            v_offset = idx * base + kv_head * HEAD_DIM

            # Load k_vec and v_vec as float32
            k_vec_raw = tl.load(k_ptr + k_offset)
            v_vec_raw = tl.load(v_ptr + v_offset)
            k_vec = k_vec_raw.to(tl.float32)  # [HEAD_DIM] f32
            v_vec = v_vec_raw.to(tl.float32)  # [HEAD_DIM] f32

            # Dot product: q_vec · k_vec
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            scaled = logits * sm_scale

            # Numerically stable LSE update
            is_init = lse == -float("inf")
            m = tl.maximum(lse, scaled)
            lse_diff = lse - scaled
            lse_update = m + tl.log(1.0 + tl.exp(-tl.abs(lse_diff)))
            lse = tl.where(is_init, scaled, lse_update)

            # Attention weight
            attn = tl.exp(scaled - lse)  # scalar

            # Accumulate output
            out_vec += attn * v_vec

        # Store results
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # out_ptr is float32; cast to bfloat16 in Python after kernel
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2)  # LSE divided by ln(2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton path (required for evaluation)
        assert TRITON_AVAILABLE, "Triton is not available. Please install triton and ensure CUDA is available."
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, "Inputs must be on CUDA for Triton."

        # Shapes and constants
        B, num_qo_heads, HEAD_DIM = q.shape
        assert num_qo_heads == 32, "Expected num_qo_heads == 32"
        assert k_cache.shape[2] == 8, "Expected num_kv_heads == 8"
        assert HEAD_DIM == 128, "Expected head_dim == 128"
        num_kv_heads = k_cache.shape[2]
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = 1.0 / math.log(2.0)

        # Compute in float32; output as bfloat16, lse as float32 divided by ln(2)
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        MAX_TOKENS = 1024  # loop bound; break when t >= num_tokens

        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr,
            output, lse,
            B=B, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM, sm_scale=float(sm_scale), gqa_ratio=gqa_ratio,
            ln2=ln2,
            MAX_TOKENS=MAX_TOKENS,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 as original model returns
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
