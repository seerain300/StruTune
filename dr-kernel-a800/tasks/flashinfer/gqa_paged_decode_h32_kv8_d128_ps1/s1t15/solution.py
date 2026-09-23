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
        out_ptr,         # *bf16, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B: tl.constexpr,               # number of batches
        num_qo_heads: tl.constexpr,    # 32
        num_kv_heads: tl.constexpr,    # 8
        HEAD_DIM: tl.constexpr,        # 128
        sm_scale: tl.constexpr,        # scaling factor (e.g., 1/sqrt(128))
        ln2_inv: tl.constexpr,         # 1 / log(2)
        GQA_RATIO: tl.constexpr,       # 4
    ):
        # One program per (b, h)
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id
        if b >= B or h >= num_qo_heads:
            return

        # Read token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q[b, h] as float32: linear offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        q_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Loop over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index into kv cache
            kv_head = h // GQA_RATIO  # GQA mapping: 32/8 = 4 groups

            # Compute offsets for k_t and v_t
            # k_ptr and v_ptr are [num_pages, 1, num_kv_heads, HEAD_DIM]
            # For fixed idx and kv_head, offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            base = idx * (num_kv_heads * HEAD_DIM)
            k_off = base + kv_head * HEAD_DIM
            v_off = base + kv_head * HEAD_DIM

            # Load k_t and v_t; assume bf16/f16 input, cast to f32 for compute
            k_t = tl.load(k_ptr + k_off).to(tl.float32)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_off).to(tl.float32)  # [HEAD_DIM]

            # Dot product: q_vec · k_t
            dot_val = tl.zeros((), dtype=tl.float32)
            for i in range(HEAD_DIM):
                dot_val += q_vec[i] * k_t[i]

            # Scale logits
            y = dot_val * sm_scale

            # Stable LSE update
            if lse == -float("inf"):
                lse = y
            else:
                diff = y - lse
                # Merge: lse_new = lse + log(1 + exp(-abs(diff))) + sign(diff) * (y - lse)
                add_term = tl.where(diff > 0.0, tl.log(1.0 + tl.exp(-diff)), tl.log(1.0 + tl.exp(diff)))
                lse = lse + add_term

            # Compute attention weight
            attn = tl.exp(y - lse)

            # Accumulate output vector
            out_vec += attn * v_t

        # Store output as bfloat16
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        out_vec_bf16 = out_vec.to(tl.bfloat16)
        tl.store(out_ptr + out_offset, out_vec_bf16)

        # Store LSE / ln(2) (float32)
        lse_scaled = lse * ln2_inv
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, \
            "All tensors must be on CUDA device for Triton kernel."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        B, num_qo_heads, HEAD_DIM = q.shape
        num_kv_heads = k_cache.shape[2]

        # Allocate outputs
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, num_qo_heads)
        ln2_inv = 1.0 / math.log(2.0)
        GQA_RATIO = num_qo_heads // num_kv_heads

        _gqa_attention_kernel[grid](
            q.to(torch.float32),                # q_ptr: float32
            k_cache, v_cache,                  # k_ptr, v_ptr: original dtype (bf16/f16)
            kv_indptr,
            output,
            lse,
            B,
            num_qo_heads,
            num_kv_heads,
            HEAD_DIM,
            sm_scale,
            ln2_inv,
            GQA_RATIO,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
