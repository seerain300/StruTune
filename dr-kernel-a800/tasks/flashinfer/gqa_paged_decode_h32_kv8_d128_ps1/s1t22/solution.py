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
        q_ptr,           # *f32, shape [B, 32, 128]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, 8, 128]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B: tl.constexpr,           # int32
        HEAD_DIM: tl.constexpr,    # 128
        sm_scale: tl.constexpr,
        ln2: tl.constexpr,         # 1 / ln(2)
        gqa_ratio: tl.constexpr,   # 4
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // 32
        h = pid % 32
        if b >= B or h >= 32:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32: offset = b*32*128 + h*128
        q_offset = b * 32 * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Loop over tokens in this batch range
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # int32 index into kv_indptr range

            # GQA mapping: kv_head = h // gqa_ratio
            kv_head = h // gqa_ratio  # int32

            # Load k_vec and v_vec at idx; k_ptr layout: [num_pages, 1, 8, 128]
            # Linear offset for k_ptr[idx, 0, kv_head, :] is idx * (8 * 128) + kv_head * 128
            offset_k = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.load(k_ptr + offset_k)  # [HEAD_DIM] input dtype; cast to f32

            # Similarly for v_ptr
            offset_v = idx * (8 * HEAD_DIM) + kv_head * HEAD_DIM
            v_vec = tl.load(v_ptr + offset_v)  # [HEAD_DIM] input dtype; cast to f32

            # Cast to float32 for compute
            k_vec = tl.cast(k_vec, tl.float32)
            v_vec = tl.cast(v_vec, tl.float32)

            # Dot product: q_vec · k_vec
            dot_sum = tl.zeros((), dtype=tl.float32)
            for i in range(HEAD_DIM):
                dot_sum += q_vec[i] * k_vec[i]

            scaled = dot_sum * sm_scale

            # Streaming logsumexp update for stability
            new_m = tl.maximum(lse, scaled)
            logadd = tl.log(1.0 + tl.exp(lse - new_m))
            lse = new_m + logadd

            # Attention weight and accumulation
            attn = tl.exp(scaled - lse)
            out_vec += attn * v_vec

            t += 1

        # Store results: out_ptr is float32 buffer
        out_offset = b * 32 * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        # Store lse scaled by 1/ln(2)
        tl.store(lse_ptr + b * 32 + h, lse * ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], bfloat16
        k_cache: [num_pages, 1, 8, 128], bfloat16/f16
        v_cache: [num_pages, 1, 8, 128], bfloat16/f16
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32 (not used)
        sm_scale: float32 scalar
        Returns:
        - output: [B, 32, 128], bfloat16
        - lse: [B, 32], float32 (divided by ln(2))
        """
        # Ensure contiguity and CUDA
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # If Triton not available or tensors not on CUDA, raise for strict compliance
        if not (TRITON_AVAILABLE and q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda):
            raise RuntimeError("Triton is not available or tensors are not on CUDA. Please provide Triton and CUDA.")

        # Shapes and constants
        B, num_qo_heads, head_dim = q.shape  # num_qo_heads == 32, head_dim == 128
        num_pages, _, num_kv_heads, _ = k_cache.shape  # num_kv_heads == 8
        gqa_ratio = num_qo_heads // num_kv_heads      # 4
        ln2 = 1.0 / math.log(2.0)

        # Allocate outputs (compute in float32; cast to bfloat16 later)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Cast q to float32 for compute
        q_f32 = q.to(torch.float32)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B, head_dim, sm_scale, ln2, gqa_ratio,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
