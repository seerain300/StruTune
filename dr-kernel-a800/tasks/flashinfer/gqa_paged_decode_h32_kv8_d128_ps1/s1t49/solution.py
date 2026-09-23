import torch
import math

# Triton availability and kernel definition
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
        k_ptr, v_ptr,    # *bf16 or *f16, shape [N, 1, 8, 128]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B,               # int32
        HEAD_DIM: tl.constexpr,          # 128
        sm_scale: tl.constexpr,          # 1/sqrt(128)
        gqa_ratio: tl.constexpr,         # 32 // 8 == 4
        ln2_inv: tl.constexpr,           # 1 / log(2)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // 32
        h = pid % 32
        if b >= B or h >= 32:
            return

        # Load q[b, h, :] as float32
        q_base = b * 32 * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base)  # [HEAD_DIM] f32

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)       # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)    # i32
        num_tokens = kv_end - kv_start             # i32 scalar

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float("inf")  # f32 scalar

        # Iterate over tokens in this batch
        for t in range(0, num_tokens):
            idx = kv_start + t  # i32

            kv_head = h // gqa_ratio  # 0..7

            # k_ptr/v_ptr layout: [N, 1, 8, 128] => linear index: idx * (1*8*128) + kv_head*128
            base = idx * (1 * 8 * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.load(k_ptr + base)  # [HEAD_DIM], may be bf16/f16; cast to f32
            v_vec = tl.load(v_ptr + base)  # [HEAD_DIM]
            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product q · k_t
            dot = tl.sum(q_vec * k_vec)  # scalar f32

            # Scale logits
            scaled = dot * sm_scale

            # Numerically stable LSE update
            new = tl.maximum(lse, scaled)
            # lse_new = new + log(1 + exp(-(new - lse)))
            lse = new + tl.log(1.0 + tl.exp(-(new - lse)))

            # Attention weight
            attn = tl.exp(scaled - lse)  # scalar

            # Accumulate output
            out_vec += attn * v_vec  # [HEAD_DIM]

        # Store output and lse (divided by ln(2))
        out_base = b * 32 * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)
        tl.store(lse_ptr + (b * 32 + h), lse * ln2_inv)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and assertions
        B, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"
        num_kv_heads = 8
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        device = q.device

        # Allocate outputs (compute in float32, return bfloat16 and float32 lse)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B=B, HEAD_DIM=head_dim, sm_scale=sm_scale, gqa_ratio=gqa_ratio, ln2_inv=1.0 / math.log(2.0),
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 as in original
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
