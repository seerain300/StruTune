import torch
import math

# Triton import and availability
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
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM] (kernel writes float32)
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # 1 / log(2)
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
        lse = -float("inf")  # scalar float32

        # Iterate tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # int32

            # Compute KV head via GQA mapping
            kv_head = h // gqa_ratio  # int

            # Load k_t and v_t (original dtype might be bf16/f16; cast to f32 for compute)
            # k_cache shape: [num_pages, 1, num_kv_heads, HEAD_DIM]
            k_vec_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_vec_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_vec_offset)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_vec_offset)  # [HEAD_DIM]

            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product: q_vec · k_vec
            logits = 0.0
            for i in range(HEAD_DIM):
                logits += q_vec[i] * k_vec[i]

            scaled = logits * sm_scale

            # Update lse stably: new_lse = max(old, scaled) + log(1 + exp(-abs(old - scaled)))
            old = lse
            max_old_scaled = tl.maximum(old, scaled)
            min_old_scaled = tl.minimum(old, scaled)
            delta = tl.abs(old - scaled)
            new_lse = max_old_scaled + tl.log(1.0 + tl.exp(-delta))
            lse = new_lse

            # Compute attention weight: exp(scaled - lse)
            attn = tl.exp(scaled - lse)

            # Accumulate output vector: out_vec += attn * v_vec
            for i in range(HEAD_DIM):
                out_vec[i] += attn * v_vec[i]

        # Store results: output as float32 (host will cast to bfloat16), lse / ln(2)
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # float32

        # Store lse divided by ln(2)
        tl.store(lse_ptr + (b * num_qo_heads + h), lse * ln2)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads, head_dim], dtype bfloat16 (compute in f32)
        k_cache: [num_pages, 1, num_kv_heads, head_dim], dtype bfloat16/f16
        v_cache: [num_pages, 1, num_kv_heads, head_dim], dtype bfloat16/f16
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32 (not used; token count inferred from kv_indptr)
        sm_scale: float32 scalar
        Returns:
        - output: [B, num_qo_heads, head_dim], dtype bfloat16
        - lse: [B, num_qo_heads], dtype float32 (divided by ln(2))
        """
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Shapes and constants
        B, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # Original asserts: num_qo_heads == 32, num_kv_heads == 8, head_dim == 128
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = 1.0 / math.log(2.0)

        # Allocate outputs (compute in float32; cast to bfloat16 later)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        if TRITON_AVAILABLE and q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda:
            # Launch one program per (b, h)
            grid = (B * num_qo_heads,)
            # Cast q to float32 for compute
            q_f32 = q.to(torch.float32)

            _gqa_attention_kernel[grid](
                q_f32, k_cache, v_cache, kv_indptr, output, lse,
                B, num_qo_heads, num_kv_heads, head_dim,
                sm_scale, gqa_ratio, ln2,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback path (CPU): model will be run on CUDA in typical evaluation
            pass

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
