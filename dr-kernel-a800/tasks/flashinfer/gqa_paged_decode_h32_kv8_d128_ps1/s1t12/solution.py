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
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *bf16, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B: tl.constexpr,
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # natural log of 2
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Token range for this batch: num_tokens = kv_indptr[b+1] - kv_indptr[b]
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32: linear offset b*HEAD_DIM*num_qo_heads + h*HEAD_DIM
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and LSE accumulator
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)
        acc = tl.full((), 0.0, dtype=tl.float32)

        # Loop over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # linear index into k/v cache
            # GQA mapping: kv_head = h // gqa_ratio
            kv_head = h // gqa_ratio
            # Linearized base for [idx, kv_head, :] in [num_pages, num_kv_heads, HEAD_DIM]
            base = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t vectors of length HEAD_DIM (cast to f32 for compute)
            k_t = tl.load(k_ptr + base)           # [HEAD_DIM] original dtype
            v_t = tl.load(v_ptr + base)           # [HEAD_DIM] original dtype

            # Compute logits = q_vec · k_t (cast k_t to f32 for dot)
            logits = tl.zeros((), dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                kd = tl.cast(k_t[d], tl.float32)
                qd = q_vec[d]  # already f32
                logits += qd * kd

            # Scale
            scaled = logits * sm_scale  # f32

            # Update LSE and acc stably
            if lse == -float("inf"):
                lse = scaled
                acc = 1.0
            else:
                m = tl.maximum(lse, scaled)
                exp1 = tl.exp(lse - m)
                exp2 = tl.exp(scaled - m)
                new_lse = m + tl.log(exp1 + exp2)
                acc = acc * tl.exp(lse - new_lse) + 1.0 * tl.exp(scaled - new_lse)
                lse = new_lse

            # Compute attention = exp(scaled - lse) and accumulate output
            attn = tl.exp(scaled - lse)
            vt = tl.cast(v_t, tl.float32)  # cast v_t to f32 for accumulation
            out_vec += attn * vt

        # Store output vector to out[b, h] as bfloat16
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        out_vec_bf16 = tl.cast(out_vec, tl.bfloat16)
        for d in range(0, HEAD_DIM):
            tl.store(out_ptr + out_offset + d, out_vec_bf16[d])

        # Store LSE / ln(2) to lse[b, h]
        lse_scaled = lse / ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available; evaluation requires Triton
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required but not available.")

        device = q.device
        # Ensure tensors are contiguous and on device
        q = q.to(device).contiguous()
        k_cache = k_cache.to(device).contiguous()
        v_cache = v_cache.to(device).contiguous()
        kv_indptr = kv_indptr.to(device).contiguous()

        B, num_qo_heads, HEAD_DIM = q.shape
        _, num_pages, num_kv_heads, _ = k_cache.shape

        # Output buffers
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q, k_cache, v_cache, kv_indptr, output, lse,
            B, num_qo_heads, num_kv_heads, HEAD_DIM, sm_scale, num_qo_heads // num_kv_heads, 1.0 / math.log(2.0)
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
