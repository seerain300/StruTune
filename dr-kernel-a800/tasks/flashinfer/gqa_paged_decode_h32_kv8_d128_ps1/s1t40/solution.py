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
        ln2: tl.constexpr,  # 1 / ln(2)
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
        num_tokens = kv_end - kv_start              # i32

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # GQA mapping: query head h maps to KV head kv_head = h // gqa_ratio
        kv_head = h // gqa_ratio  # int

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float('inf')  # f32 scalar

        # Iterate over tokens in this batch
        for t in range(0, num_tokens):
            idx = kv_start + t  # i32, token index in [0, total_tokens)

            # Compute offsets for k/v vectors at (idx, 0, kv_head, :)
            # k_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM]
            # v_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM]
            # Since the middle dim is 1, stride for kv_head is HEAD_DIM.
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t (dtype of k_ptr/v_ptr can be bf16/f16; we cast to f32 for compute)
            k_vec = tl.load(k_ptr + k_offset)  # [HEAD_DIM], cast to f32 (happens outside kernel in Python)
            v_vec = tl.load(v_ptr + v_offset)  # [HEAD_DIM], cast to f32

            # Dot product: q_vec · k_vec
            dot = 0.0
            for d in range(0, HEAD_DIM):
                dot += q_vec[d] * k_vec[d]

            # Scale logits
            scaled = dot * sm_scale  # f32

            # Update LSE stably: if lse == -inf, set lse = scaled; else lse = max(lse, scaled) + log(1 + exp(scaled - lse))
            if lse == -float('inf'):
                lse = scaled
            else:
                maxv = tl.maximum(lse, scaled)
                minv = tl.minimum(lse, scaled)
                lse = maxv + tl.log(1.0 + tl.exp(minv - maxv))

            # Attention weight
            attn = tl.exp(scaled - lse)  # f32 scalar

            # Accumulate output vector
            for d in range(0, HEAD_DIM):
                out_vec[d] += attn * v_vec[d]

        # Store results
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # out_ptr is float32

        # Store LSE divided by ln(2)
        lse_div = lse * ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_div)  # lse_ptr is float32


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B, num_qo_heads, HEAD_DIM = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and HEAD_DIM == 128, "Fixed constants expected"

        # Ensure contiguity and device
        q_f32 = q.to(torch.float32).contiguous()  # compute in float32 for stability
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Allocate outputs (compute in f32, cast to bfloat16 at the end)
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Grid: one program per (b, h)
        grid = (B * num_qo_heads,)

        # Launch Triton kernel
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM,
            sm_scale=float(sm_scale),
            gqa_ratio=num_qo_heads // num_kv_heads,  # 4
            ln2=1.0 / math.log(2.0),                # ~1.44269504
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
