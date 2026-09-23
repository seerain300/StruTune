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
        k_ptr, v_ptr,    # *bf16 or *f16, shape [NUM_PAGES, 1, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM] (we'll cast to bf16 in host)
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B: tl.constexpr,         # batch size (constexpr for indexing simplicity)
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,       # 1 / log(2) for division
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Compute token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)       # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)    # i32
        num_tokens = kv_end - kv_start             # i32 scalar

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)          # [HEAD_DIM] f32

        # Initialize output vector and running LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        m = -float('inf')                          # running max of scaled logits
        lse = -float('inf')                       # running logsumexp of scaled logits

        # Loop over tokens
        # Note: Triton supports for-loops with dynamic bounds; num_tokens is scalar
        for t in range(0, num_tokens):
            idx = kv_start + t                    # token index into cache
            # Compute KV head for GQA
            kv_head = h // gqa_ratio             # int scalar

            # Compute linear offsets for k_ptr/v_ptr:
            # k_ptr layout: [NUM_PAGES, 1, NUM_KV_HEADS, HEAD_DIM]
            # For fixed kv_head, we pick one of 8 groups; the "1" dimension is always 0.
            # Addressing: ptr + page * (1*NUM_KV_HEADS*HEAD_DIM) + kv_head * HEAD_DIM + t * HEAD_DIM
            # But since dim-1 is HEAD_DIM, we can directly use:
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_offset, mask=True)    # [HEAD_DIM], source dtype will be cast automatically when used in f32 ops
            v_vec = tl.load(v_ptr + v_offset, mask=True)    # [HEAD_DIM]

            # Cast to float32 for compute
            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product: q_vec · k_vec
            # Compute sum over elements
            dot = tl.sum(q_vec * k_vec, axis=0)

            scaled = dot * sm_scale

            # Streaming stable LSE update:
            # If scaled > m: new_lse = lse + (scaled - m) + log(1 + exp(m - scaled))
            # Else: new_lse = lse + log(1 + exp(scaled - m))
            # And m = max(m, scaled).
            greater = scaled > m
            new_lse = tl.where(greater,
                               lse + (scaled - m) + tl.log(1.0 + tl.exp(m - scaled)),
                               lse + tl.log(1.0 + tl.exp(scaled - m)))
            m = tl.maximum(m, scaled)
            lse = new_lse

            attn = tl.exp(scaled - lse)

            out_vec += attn * v_vec

        # Store output vector (float32) and LSE / ln(2) to lse
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_bh = lse * ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_bh)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA if Triton is available; otherwise, we can still run with CPU tensors
        # The evaluation environment will place inputs on CUDA; this forward assumes CUDA tensors.
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, "Inputs must be CUDA tensors for Triton kernel"

        # Shapes
        B, num_qo_heads, HEAD_DIM = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # GQA ratio
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2_inv = 1.0 / math.log(2.0)

        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Output tensors
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)  # compute in f32
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)                # store lse / ln(2)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32,
            k_cache, v_cache,
            kv_indptr,
            output,
            lse,
            B,
            num_qo_heads,
            num_kv_heads,
            HEAD_DIM,
            sm_scale,
            gqa_ratio,
            ln2_inv,
            num_warps=4, num_stages=2,
        )

        # Cast output to bfloat16 as expected by original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse  # lse is float32


def run(*args):
    return ModelNew()(*args)
