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

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32: q_ptr[b, h, :]
        q_base = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Iterate over tokens in [kv_start, kv_start + num_tokens)
        # We loop explicitly in Triton with a Python-like for: Triton allows simple loops with in-kernel scalars.
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index

            # Compute corresponding KV head (GQA)
            kv_head = h // (num_qo_heads // num_kv_heads)  # == h // 4

            # Gather k_t and v_t: k_ptr[idx, kv_head, :]
            # Assuming k_ptr layout [num_pages, num_kv_heads, HEAD_DIM]
            # Address: idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_offsets = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offsets = k_offsets  # same offsets for v_ptr

            k_t = tl.load(k_ptr + k_offsets + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], original dtype (cast later)
            v_t = tl.load(v_ptr + v_offsets + tl.arange(0, HEAD_DIM))

            # Cast to float32 for compute
            k_t = k_t.to(tl.float32)
            v_t = v_t.to(tl.float32)

            # Dot product: q_vec · k_t
            logits = tl.sum(q_vec * k_t, axis=0)  # scalar
            scaled = logits * sm_scale

            # Update LSE stably
            # If lse == -inf: lse = scaled
            # Else: lse = lse + log(1 + exp(scaled - lse))
            is_inf = lse == -float("inf")
            # Compute new_lse
            if is_inf:
                new_lse = scaled
            else:
                delta = scaled - lse
                new_lse = lse + tl.log(1.0 + tl.exp(delta))

            # Compute attention weight
            attn = tl.exp(scaled - new_lse)  # after update, scaled becomes new_lse for next iteration

            # Accumulate output
            out_vec += attn * v_t

            # Update lse for next iteration
            lse = new_lse

        # Store results
        out_base = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_base + tl.arange(0, HEAD_DIM), out_vec)

        # Store LSE divided by ln(2)
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Shapes
        B, num_qo_heads, HEAD_DIM = q_f32.shape
        num_pages, num_kv_heads, _, _ = k_cache.shape
        # Allocate outputs (compute in f32, cast to bf16 later)
        output_f32 = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q_f32.device)
        lse_f32 = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q_f32.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        ln2 = 1.0 / math.log(2.0)

        _gqa_attention_kernel[grid](
            q_f32,
            k_cache, v_cache,
            kv_indptr,
            output_f32,
            lse_f32,
            B,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM,
            sm_scale=sm_scale,
            ln2=ln2,
        )

        # Cast output to bfloat16 as required by original
        output = output_f32.to(torch.bfloat16)
        lse = lse_f32  # already float32
        return output, lse


def run(*args):
    return ModelNew()(*args)
