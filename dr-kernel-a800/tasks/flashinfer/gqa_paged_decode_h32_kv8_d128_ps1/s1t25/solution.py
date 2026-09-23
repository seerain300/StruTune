import torch
import math

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Define Triton kernel for attention (used when Triton + CUDA available)
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
        ln2_inv: tl.constexpr,  # 1 / ln(2)
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

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float('inf')  # f32

        # Iterate over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index

            # GQA mapping: kv_head = h // 4
            kv_head = h // gqa_ratio  # int32

            # Compute linear offsets into k_ptr/v_ptr:
            # Flattened length = num_pages * num_kv_heads * HEAD_DIM
            # Index for (idx, kv_head): idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            linear = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t (assume bfloat16 or float16 in inputs), cast to float32
            k_t_bf16 = tl.load(k_ptr + linear)  # [HEAD_DIM] bf16/f16
            v_t_bf16 = tl.load(v_ptr + linear)  # [HEAD_DIM] bf16/f16
            k_t = k_t_bf16.to(tl.float32)
            v_t = v_t_bf16.to(tl.float32)

            # Dot product
            logits = tl.sum(q_vec * k_t, axis=0)  # scalar f32
            scaled = logits * sm_scale  # f32

            # Stable LSE update: new_lse = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            max_ls = tl.maximum(lse, scaled)
            diff = lse - scaled
            exp_term = tl.exp(-tl.abs(diff))
            add_term = tl.log(1.0 + exp_term)
            new_lse = max_ls + add_term
            lse = new_lse

            # Attention
            attn = tl.exp(scaled - lse)  # scalar f32

            # Accumulate
            out_vec += attn * v_t  # [HEAD_DIM] f32

        # Store results
        out_offset = (b * num_qo_heads + h) * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # float32

        # Store lse / ln(2) = lse * (1 / ln(2))
        lse_scaled = lse * ln2_inv
        tl.store(lse_ptr + (b * num_qo_heads + h), lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape

        # Validate shapes (as in original)
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Expected q[*,32,128], k/v[*,1,8,128]"
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2_inv = 1.0 / math.log(2.0)

        # If Triton + CUDA available, use GPU kernel; else CPU fallback
        use_triton = TRITON_AVAILABLE and torch.cuda.is_available()

        if use_triton:
            # Move inputs to GPU for Triton computation
            q_gpu = q.to(torch.float32)  # compute in f32
            k_gpu = k_cache  # keep original dtype (bf16/f16); cast in-kernel
            v_gpu = v_cache  # keep original dtype; cast in-kernel
            kv_indptr_gpu = kv_indptr  # int32

            output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q_gpu.device)
            lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q_gpu.device)

            # Launch Triton kernel: one program per (b, h)
            grid = (batch_size * num_qo_heads,)
            _gqa_attention_kernel[grid](
                q_gpu,
                k_gpu, v_gpu,
                kv_indptr_gpu,
                output,
                lse,
                batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                HEAD_DIM=head_dim,
                sm_scale=float(sm_scale),
                gqa_ratio=gqa_ratio,
                ln2_inv=float(ln2_inv),
            )

            # Cast output to bfloat16 as in original
            output_bf16 = output.to(torch.bfloat16)
            return output_bf16, lse
        else:
            # CPU fallback: pure PyTorch implementation mirroring original logic
            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device
            )

            for b in range(batch_size):
                kv_start = int(kv_indptr[b].item())


def run(*args):
    return ModelNew()(*args)
