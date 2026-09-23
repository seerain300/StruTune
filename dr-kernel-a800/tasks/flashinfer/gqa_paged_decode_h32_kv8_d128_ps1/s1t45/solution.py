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
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B,               # int32
        HEAD_DIM: tl.constexpr,          # 128
        sm_scale: tl.constexpr,          # 1/sqrt(128)
        gqa_ratio: tl.constexpr,         # 32//8 == 4
        ln2: tl.constexpr,               # 1/log(2)
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

        # Compute number of tokens for this batch: tokens[b] = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
        kv_start = tl.load(kv_indptr_ptr + b)       # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)    # i32
        num_tokens = kv_end - kv_start             # i32 scalar

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float("inf")  # f32 scalar

        # Iterate over tokens in this batch
        # Layout for k_ptr/v_ptr: [num_pages, 1, num_kv_heads, HEAD_DIM]
        # For each token index idx, kv_head = h // gqa_ratio
        for t in range(0, num_tokens):
            idx = kv_start + t  # i32

            kv_head = h // gqa_ratio  # 0..7

            # Compute base for k/v at idx and kv_head
            k_base = idx * (1 * 8 * HEAD_DIM) + kv_head * HEAD_DIM  # 1 is the dummy middle dim
            k_vec = tl.load(k_ptr + k_base)  # [HEAD_DIM], may be bf16/f16; cast to f32
            v_vec = tl.load(v_ptr + k_base)  # [HEAD_DIM]

            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product q · k_t
            dot = tl.zeros((), dtype=tl.float32)
            for i in range(HEAD_DIM):
                dot += q_vec[i] * k_vec[i]

            # Scale logits
            scaled = dot * sm_scale  # f32 scalar

            # Numerically stable LSE update
            max_lse_scaled = tl.maximum(lse, scaled)
            delta = tl.abs(lse - scaled)
            lse = max_lse_scaled + tl.log(1.0 + tl.exp(-delta))

            # Attention weight
            attn = tl.exp(scaled - lse)  # f32 scalar

            # Accumulate output vector
            out_vec += attn * v_vec  # elementwise

        # Store results
        out_base = b * (32 * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)
        tl.store(lse_ptr + b * 32 + h, lse / ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32 and head_dim == 128, "q must have shape [B, 32, 128]"
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8, "k_cache/v_cache must have 8 KV heads"

        # Allocate outputs (compute in f32, output in bfloat16; lse in f32)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        q_f32 = q.to(torch.float32)

        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            batch_size,
            HEAD_DIM=head_dim,
            sm_scale=sm_scale,
            gqa_ratio=num_qo_heads // num_kv_heads,  # 4
            ln2=1.0 / math.log(2.0),
            num_warps=4,
        )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
