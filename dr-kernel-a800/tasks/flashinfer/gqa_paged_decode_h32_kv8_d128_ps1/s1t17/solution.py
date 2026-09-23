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
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM] (we'll cast to bf16 in host)
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        ln2_inv: tl.constexpr,  # 1 / ln(2)
        GQA_RATIO: tl.constexpr,  # num_qo_heads // num_kv_heads = 4
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

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float('inf')

        # Loop over tokens in this batch
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index into k/v arrays

            # GQA head mapping
            kv_head = h // GQA_RATIO

            # Compute linear offsets for k_ptr/v_ptr:
            # k_ptr/v_ptr shape is [num_pages, 1, num_kv_heads, HEAD_DIM]
            # Accessing [idx, 0, kv_head, :] has linear offset:
            kv_head_offset = kv_head * HEAD_DIM
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head_offset
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head_offset

            # Load k_vec and v_vec (original dtype); cast to f32 for compute
            k_vec = tl.load(k_ptr + k_offset)
            v_vec = tl.load(v_ptr + v_offset)
            k_vec_f32 = k_vec.to(tl.float32)
            v_vec_f32 = v_vec.to(tl.float32)

            # Dot product q·k
            logits = 0.0
            for d in range(HEAD_DIM):
                logits += q_vec[d] * k_vec_f32[d]

            # Scale logits
            scaled = logits * sm_scale

            # Update LSE stably
            if lse == -float('inf') or scaled > lse:
                lse = scaled + tl.log(1.0 + tl.exp(lse - scaled))
            else:
                lse = lse + tl.log(1.0 + tl.exp(scaled - lse))

            # Compute attention weight
            attn = tl.exp(scaled - lse)

            # Accumulate output
            out_vec += attn * v_vec_f32

        # Store results
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)             # float32 output (will cast to bf16 on host)
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2_inv)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."
        B, num_qo_heads, HEAD_DIM = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and HEAD_DIM == 128, "Expected shapes: num_qo_heads=32, num_kv_heads=8, head_dim=128."
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have length B+1."

        # Allocate outputs (compute in float32, cast later)
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # GQA ratio and constants
        GQA_RATIO = 4
        ln2_inv = 1.0 / math.log(2.0)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q.to(torch.float32).contiguous(),              # q_ptr: float32
            k_cache.contiguous(), v_cache.contiguous(),   # k_ptr, v_ptr: original dtype
            kv_indptr.contiguous(),
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

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        # lse already stored as lse / ln(2)
        return output, lse


def run(*args):
    return ModelNew()(*args)
