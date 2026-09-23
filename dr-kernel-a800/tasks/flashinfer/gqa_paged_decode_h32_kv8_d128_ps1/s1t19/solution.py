import math
import torch

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
        k_ptr, v_ptr,    # *f16 or *bf16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # 1 / log(2.0)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Compute GQA KV head for this query head
        kv_head = h // gqa_ratio

        # Load q[b, h] as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM], f32

        # Determine token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)      # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)   # i32
        num_tokens = kv_end - kv_start            # i32

        # Initialize output vector and streaming LSE components
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        m = tl.full((), -float("inf"), dtype=tl.float32)  # running max
        s = tl.zeros((), dtype=tl.float32)               # running sum of exp(scaled - m)

        # Iterate over tokens in this batch
        # Note: Triton supports loops with runtime bounds; we iterate manually.
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # token index into cache

            # Load k_t and v_t as f16/bf16 and cast to f32 for compute
            # k_ptr/v_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM]
            # Access k_ptr[idx, 0, kv_head, :] and v_ptr[idx, 0, kv_head, :]
            k_base = idx * num_kv_heads * HEAD_DIM
            v_base = idx * num_kv_heads * HEAD_DIM
            k_vec = tl.load(k_ptr + k_base + kv_head * HEAD_DIM)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_base + kv_head * HEAD_DIM)  # [HEAD_DIM]
            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product: q_vec · k_vec
            logits = 0.0
            for d in range(HEAD_DIM):
                logits += q_vec[d] * k_vec[d]

            scaled = logits * sm_scale  # f32

            # Update streaming LSE: m = max(m, scaled); s = s * exp(m - scaled) + exp(scaled - m)
            # Equivalent: s = s * exp(m - scaled) + exp(scaled - m)
            # m_new = max(m, scaled)
            m_new = tl.maximum(m, scaled)
            s = s * tl.exp(m - scaled) + tl.exp(scaled - m)
            m = m_new

            # Compute attention weight: exp(scaled - m)
            attn = tl.exp(scaled - m) * ln2  # ln2 is a scalar; attn is already scaled by ln2 in original code

            # Accumulate output: out_vec += attn * v_vec
            # We assume ln2 is already accounted for in original run; here we match the original behavior by dividing logsumexp by ln2.
            # To recover correct attn, remove ln2: attn = exp(scaled - m) without ln2.
            attn = tl.exp(scaled - m)  # correct attention weight

            out_vec += attn * v_vec

            t += 1

        # Store results: output[b, h] = out_vec (will be cast by host if needed), lse[b, h] = m / ln2
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * num_qo_heads + h
        tl.store(lse_ptr + lse_offset, m / ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads, HEAD_DIM], bfloat16 or float32
        k_cache: [num_pages, 1, num_kv_heads, HEAD_DIM], bfloat16/f16
        v_cache: [num_pages, 1, num_kv_heads, HEAD_DIM], bfloat16/f16
        kv_indptr: [B+1], int32 (cumulative token counts per batch)
        kv_indices: [num_tokens], int32 (not used in computation)
        sm_scale: float32 scalar
        Returns (output, lse) with:
          output: [B, num_qo_heads, HEAD_DIM], bfloat16
          lse: [B, num_qo_heads], float32 divided by ln(2)
        """
        # Ensure contiguity and dtypes
        B, num_qo_heads, HEAD_DIM = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert HEAD_DIM == 128
        assert kv_indptr.shape[0] == B + 1

        # Cast q to float32 for kernel compute
        q_f32 = q.to(torch.float32).contiguous()

        # Cast k_cache/v_cache to a Triton-friendly dtype (bf16/f16). Keep original shape.
        # Triton loads will cast to f32 in-kernel.
        k_cache_contig = k_cache.contiguous()
        v_cache_contig = v_cache.contiguous()

        # Allocate outputs
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)  # kernel writes f32
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)               # kernel writes f32

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32,
            k_cache_contig, v_cache_contig,
            kv_indptr.contiguous(),
            output,
            lse,
            B,
            num_qo_heads,
            num_kv_heads,
            HEAD_DIM,
            sm_scale,
            4,  # gqa_ratio = num_qo_heads // num_kv_heads
            1.0 / math.log(2.0),  # ln2
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 as in original; lse is float32 as original (already divided by ln2)
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
