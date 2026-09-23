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
        k_ptr, v_ptr,    # *f16 or *bf16, shape [num_pages, num_kv_heads, HEAD_DIM]
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

        # Determine token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32
        q_base = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))  # [HEAD_DIM] f32

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float('inf')

        # Iterate tokens: idx = kv_start + t
        # Note: We cannot use a Python for-loop with dynamic bounds; use a while loop.
        t = 0
        while t < num_tokens:
            idx = kv_start + t
            # Compute GQA mapping: kv_head = h // (num_qo_heads // num_kv_heads) = h // 4
            kv_head = h // (num_qo_heads // num_kv_heads)

            # Compute linear offsets for k/v: k_ptr/v_ptr are [num_pages, num_kv_heads, HEAD_DIM]
            # We index by (idx, kv_head) assuming kv_indptr encodes token indices into the first dim.
            k_offset = idx * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM
            v_offset = k_offset  # same offset

            # Load k_t and v_t (original dtype may be f16/bf16), cast to f32 for compute
            k_t = tl.load(k_ptr + k_offset + tl.arange(0, HEAD_DIM))
            v_t = tl.load(v_ptr + v_offset + tl.arange(0, HEAD_DIM))
            k_t = k_t.to(tl.float32)
            v_t = v_t.to(tl.float32)

            # Dot product: q_vec · k_t
            logits = tl.sum(q_vec * k_t, axis=0)  # scalar f32

            # Scale logits
            scaled = logits * sm_scale  # f32 scalar

            # Streaming, numerically stable LSE update
            # if lse == -inf: lse = scaled
            # else: lse = lse + log(1 + exp(scaled - lse))
            is_neg_inf = lse == -float('inf')
            new_lse = tl.where(
                is_neg_inf,
                scaled,
                lse + tl.log(1.0 + tl.exp(scaled - lse))
            )
            lse = new_lse

            # Attention weight
            attn = tl.exp(scaled - lse)  # scalar f32

            # Accumulate output vector: out_vec += attn * v_t
            out_vec += attn * v_t

            t += 1

        # Store output vector and LSE / ln(2)
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset + tl.arange(0, HEAD_DIM), out_vec)
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available and inputs are on CUDA
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, "Inputs must be CUDA tensors"

        # Shapes and constants
        B, num_qo_heads, HEAD_DIM = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        gqa_ratio = num_qo_heads // num_kv_heads
        assert num_qo_heads == 32 and num_kv_heads == 8 and HEAD_DIM == 128, "Fixed shape constraints expected"

        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Allocate outputs (float32 compute, cast to bfloat16 later)
        output_f32 = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        ln2 = 1.0 / math.log(2.0)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output_f32, lse_f32,
            B, num_qo_heads, num_kv_heads, HEAD_DIM,
            sm_scale, ln2,
            num_warps=4,
        )

        # Return output as bfloat16 and lse as float32
        output_bf16 = output_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
