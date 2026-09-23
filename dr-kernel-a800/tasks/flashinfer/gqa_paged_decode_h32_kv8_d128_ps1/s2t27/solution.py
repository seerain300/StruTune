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
        q_ptr,                 # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,                 # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,                 # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,         # *int32, shape [BATCH_SIZE + 1]
        kv_indices_ptr,        # *int32, shape [NUM_KV_INDICES]
        out_ptr,               # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,               # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,              # float32 scalar
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,          # upper bound on tokens per batch (e.g., 1<<20)
        HALF_LN2_INV: tl.constexpr,      # 1 / ln(2) = 1.442695...
    ):
        # program ids: one program per (batch, query head)
        b = tl.program_id(0)  # int
        h = tl.program_id(1)  # int

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Determine token range for this batch
        start = tl.load(kv_indptr_ptr + b)              # int32
        end = tl.load(kv_indptr_ptr + b + 1)           # int32
        num_tokens_actual = end - start                # int32

        # Load q[h] vector
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Pass 1: compute max and sum(exp(s - max)) across tokens
        max_s = tl.full((), -1.0e20, tl.float32)  # scalar
        sum_exp = tl.zeros((), tl.float32)        # scalar
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32 token index

            # Offsets for k and v rows
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp += tl.exp(s - max_s)

        # lse = log(sum_exp) / ln(2) == log(sum_exp) * (1/ln(2))
        lse_val = tl.log(sum_exp) * HALF_LN2_INV

        # Store lse[b, h]
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: compute output vector out[b, h, :] = sum_i attn_i * v_i, where attn_i = exp(s - lse_val)
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)

            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale

            if mask_i:
                attn = tl.exp(s - lse_val)
                out_vec += attn * v_vec

        # Store output[b, h, :]
        tl.store(out_ptr + b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton availability
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        # Device and dtype handling
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be CUDA tensors for Triton execution"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16, \
            "Inputs must be bfloat16 as per original code"
        assert kv_indptr.dtype == torch.int32 and kv_indices.dtype == torch.int32

        # Shapes
        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        num_pages = k_cache.shape[0]
        num_kv_heads = k_cache.shape[2]

        # Constants
        NUM_QO_HEADS = num_qo_heads  # 32
        NUM_KV_HEADS = num_kv_heads  # 8
        HEAD_DIM = head_dim          # 128
        # Upper bound for tokens per batch: large to cover all workloads
        NUM_TOKS = 1 << 20           # 1,048,576
        HALF_LN2_INV = 1.4426950408889634  # 1 / ln(2)

        # Make inputs contiguous and cast to float32 for compute
        q32 = q.contiguous().to(torch.float32)
        k32 = k_cache.contiguous().to(torch.float32)
        v32 = v_cache.contiguous().to(torch.float32)
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Allocate output and lse
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        _attention_bh_kernel[grid](
            q32, k32, v32, kv_indptr, kv_indices, output, lse, sm_scale,
            batch_size, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, NUM_TOKS, HALF_LN2_INV,
            num_warps=4, num_stages=2
        )

        # Return output in bfloat16 (to match original), and lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
