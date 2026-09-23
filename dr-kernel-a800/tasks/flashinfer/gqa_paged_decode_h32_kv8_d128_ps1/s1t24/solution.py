import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _gqa_attention_kernel(
        q_ptr,           # *f32, shape [B, 32, 128]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, 8, 128]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B: tl.constexpr,                   # batch size (compile-time for Triton grid)
        HEAD_DIM: tl.constexpr,           # 128
        sm_scale: tl.constexpr,           # scaling factor
        gqa_ratio: tl.constexpr,          # 4
        ln2: tl.constexpr                  # 1 / log(2.0)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // 32
        h = pid % 32
        if b >= B or h >= 32:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)     # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)  # i32
        num_tokens = kv_end - kv_start           # i32

        # Load q vector for this (b, h) as float32
        q_offset = b * 32 * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float('inf'), dtype=tl.float32)

        # Iterate tokens in this batch
        # Note: num_tokens is a runtime scalar; Triton supports while loops.
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # token index into cache

            # Compute kv head mapping for GQA
            kv_head = h // gqa_ratio  # 4

            # Load k and v vectors from cache, cast to f32
            k_offset = idx * (1 * 8 * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.load(k_ptr + k_offset)  # [HEAD_DIM] (bf16/f16), Triton will load raw; cast below
            v_offset = idx * (1 * 8 * HEAD_DIM) + kv_head * HEAD_DIM
            v_vec = tl.load(v_ptr + v_offset)  # [HEAD_DIM] (bf16/f16), cast below

            # Cast to float32 for compute
            k_vec = tl.cast(k_vec, tl.float32)
            v_vec = tl.cast(v_vec, tl.float32)

            # Dot product: q_vec · k_vec
            # Manually compute sum(q * k) across HEAD_DIM
            dot = 0.0
            d = 0
            while d < HEAD_DIM:
                dot += q_vec[d] * k_vec[d]
                d += 1

            scaled = dot * sm_scale

            # Streaming logsumexp update
            new_m = tl.maximum(lse, scaled)
            lse = new_m + tl.log(1.0 + tl.exp(lse - new_m))

            # Attention weight and accumulation
            attn = tl.exp(scaled - lse)
            out_vec += attn * v_vec

            t += 1

        # Store output vector and LSE / ln(2)
        out_offset = b * 32 * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)
        tl.store(lse_ptr + b * 32 + h, lse * ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        B, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32 and head_dim == 128
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = 1.0 / math.log(2.0)

        # Allocate outputs (compute in float32; cast to bfloat16 at the end)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Cast q to float32 for compute
        q_f32 = q.to(torch.float32)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B, head_dim, sm_scale, gqa_ratio, ln2,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
