import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, [NUM_KV_INDICES]
        out_ptr,          # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,         # float32 scalar
        half_ln2_inv,     # float32 scalar: 1 / ln(2)
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping: kv_head = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio

        # Base offset for q[b, h, :]
        q_base = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM

        # First pass: compute max and sum(exp(s - max)) across tokens
        max_s = -1e20  # scalar float32
        sum_exp = 0.0  # scalar float32

        for i in range(NUM_TOKS):
            mask_i = i < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
            # token index for this i
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b] + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            # Load q_vec [HEAD_DIM]
            q_vec = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))
            # Load k_vec and v_vec [HEAD_DIM]
            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)

            # Dot product scalar
            dot_i = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s_i = dot_i * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s_i)
                sum_exp = sum_exp * tl.exp(sum_exp - s_i) + 1.0  # keep sum_exp = sum(exp(s - max_s)) when s_i > max_s

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv
        # Store lse for this (b, h)
        lse_base = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Second pass: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b] + i, mask=mask_i, other=0)

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            q_vec = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))
            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)

            dot_i = tl.sum(q_vec * k_vec, axis=0)
            s_i = dot_i * sm_scale
            attn_i = tl.exp(s_i - lse_val)  # softmax contribution
            out_vec += attn_i * v_vec  # accumulate vector

        # Store output for this (b, h)
        tl.store(out_ptr + q_base, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        # Ensure CUDA and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Cast to float32 for computation
        q = q.to(torch.float32)
        k_cache = k_cache.to(torch.float32)
        v_cache = v_cache.to(torch.float32)

        batch_size, num_qo_heads, head_dim = q.shape
        _, num_pages, num_kv_heads, _ = k_cache.shape
        len_indptr = kv_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Output and lse tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Grid: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # Launch Triton kernel
        _attention_bh_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices,
            output, lse,
            float(sm_scale), float(1.0 / math.log(2.0)),
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=1024,  # large bound; masked by actual num tokens
            num_warps=4,
            num_stages=2,
        )

        # Return output as bfloat16 (to match original) and lse as float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
