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
    def _attention_bh_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM], contiguous
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM], contiguous
        kv_indptr_ptr,   # *i32, shape [B+1], contiguous
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM], contiguous
        lse_ptr,         # *f32, shape [B, num_qo_heads], contiguous
        B: tl.constexpr,             # batch size
        num_qo_heads: tl.constexpr,  # 32
        num_kv_heads: tl.constexpr,  # 8
        HEAD_DIM: tl.constexpr,      # 128
        sm_scale: tl.constexpr,      # float32 scalar
        ln2: tl.constexpr,           # 1 / ln(2) as float32
        gqa_ratio: tl.constexpr      # num_qo_heads // num_kv_heads, here 4
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)       # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)    # i32
        num_tokens = kv_end - kv_start             # i32

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)          # [HEAD_DIM] f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float("inf")                        # f32 scalar

        t = 0
        while t < num_tokens:
            idx = kv_start + t                     # int32
            # GQA mapping: kv_head = h // gqa_ratio
            kv_head = h // gqa_ratio              # int

            # Load k_t and v_t (bf16/f16) as float32 vectors
            k_offset = idx * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM
            v_offset = idx * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM
            k_t = tl.load(k_ptr + k_offset)       # [HEAD_DIM], f16/bf16 -> cast in-kernel to f32
            v_t = tl.load(v_ptr + v_offset)       # [HEAD_DIM], same

            # Cast to f32 for compute
            k_t = tl.cast(k_t, tl.float32)
            v_t = tl.cast(v_t, tl.float32)

            # Dot product: q_vec · k_t
            # Implement as sum over elements
            dot = 0.0
            for i in range(HEAD_DIM):
                dot += q_vec[i] * k_t[i]

            scaled = dot * sm_scale                # f32 scalar

            # Numerically stable LSE update: new_lse = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            diff = scaled - lse
            new_lse = tl.maximum(lse, scaled) + tl.log(1.0 + tl.exp(-tl.abs(diff)))
            lse = new_lse

            attn = tl.exp(scaled - lse)           # attention for this token
            out_vec += attn * v_t                 # accumulate contribution

            t += 1

        # Store output and lse
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        # kv_indices is not used in original logic, so we ignore it

        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        # Allocate outputs: compute in f32, return out as bfloat16 and lse as float32 / ln(2)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        ln2 = 1.0 / math.log(2.0)
        gqa_ratio = num_qo_heads // num_kv_heads

        _attention_bh_kernel[grid](
            q.to(torch.float32),   # q must be float32 for compute
            k_cache, v_cache,      # original dtypes, cast to f32 in-kernel
            kv_indptr,
            output,
            lse,
            B=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            sm_scale=sm_scale,
            ln2=ln2,
            gqa_ratio=gqa_ratio,
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
