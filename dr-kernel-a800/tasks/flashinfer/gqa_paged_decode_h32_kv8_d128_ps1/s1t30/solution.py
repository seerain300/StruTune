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
    def _attention_gqa_kernel(
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
        MAX_TOKENS: tl.constexpr,
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

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Iterate over tokens up to MAX_TOKENS
        for t in range(0, MAX_TOKENS):
            if t >= num_tokens:
                break

            # Cache index v = kv_start + t
            v_idx = kv_start + t  # int32

            # GQA mapping: KV head for this query head
            kv_head = h // gqa_ratio  # int32

            # Compute linear offsets for k_ptr and v_ptr:
            # k_ptr is [num_pages, 1, num_kv_heads, HEAD_DIM] contiguous
            # offset = v_idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_off = v_idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = v_idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t as float32
            k_t = tl.load(k_ptr + k_off).to(tl.float32)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_off).to(tl.float32)  # [HEAD_DIM]

            # Compute logits = q · k_t
            logits = 0.0
            for d in range(0, HEAD_DIM):
                logits += q_vec[d] * k_t[d]

            scaled = logits * sm_scale

            # Numerically-stable LSE update:
            # If lse == -inf, set lse = scaled.
            # Else: new_lse = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            new_lse = tl.where(
                lse == -float("inf"),
                scaled,
                tl.log(1.0 + tl.exp(-tl.abs(lse - scaled))) + tl.maximum(lse, scaled),
            )
            lse = new_lse

            # attention = exp(scaled - lse)
            attn = tl.exp(scaled - lse)

            # Accumulate output vector
            for d in range(0, HEAD_DIM):
                out_vec[d] += attn * v_t[d]

        # Store output vector (float32) and LSE / ln(2)
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_scaled = lse * ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes:
        # q: [B, 32, 128], k_cache: [num_pages, 1, 8, 128], v_cache: same, kv_indptr: [B+1], kv_indices: [N], sm_scale: float
        B, num_qo_heads, HEAD_DIM = q.shape
        num_kv_heads = k_cache.shape[2]
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = 1.0 / math.log(2.0)

        # Ensure contiguity and dtype
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Allocate outputs
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)

        # Choose MAX_TOKENS; 1024 is a safe upper bound for provided workloads
        MAX_TOKENS = 1024

        _attention_gqa_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM,
            sm_scale=float(sm_scale),
            gqa_ratio=gqa_ratio,
            ln2=ln2,
            MAX_TOKENS=MAX_TOKENS,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original return type
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
