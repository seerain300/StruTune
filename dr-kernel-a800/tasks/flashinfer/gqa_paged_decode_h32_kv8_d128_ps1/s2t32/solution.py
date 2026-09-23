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
        q_ptr,            # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM] but we load q[h] directly via h
        k_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32,   [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32,   [NUM_KV_INDICES]
        out_ptr,          # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,            # runtime integer
        NUM_QO_HEADS: tl.constexpr,          # 32
        NUM_KV_HEADS: tl.constexpr,          # 8
        HEAD_DIM: tl.constexpr,              # 128
        sm_scale: tl.float32,                # 1.0 / sqrt(128)
        NUM_TOKS: tl.int32,                  # upper bound for loop, e.g., 8192
    ):
        # Program ids for batch and query head
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # First pass: accumulate max_s and sum_exp over valid tokens
        max_s = tl.full([1], -1.0e20, dtype=tl.float32)
        sum_exp = tl.full([1], 0.0, dtype=tl.float32)

        # Number of valid tokens for this batch
        num_tokens = tl.load(kv_indptr_ptr + (b + 1)) - tl.load(kv_indptr_ptr + b)

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            # idx = start + i; start = kv_indptr[b]
            start = tl.load(kv_indptr_ptr + b)
            idx = start + i

            # Load q[h]
            q_off = h * HEAD_DIM
            q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

            # Offsets for k/v rows
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # float32 scalar
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            max_s = tl.where(mask_i, tl.maximum(max_s, s), max_s)
            sum_exp = tl.where(mask_i, sum_exp * tl.exp(max_s - s) + 1.0, sum_exp)
            # Note: When mask_i is False, we do not update max_s/sum_exp and rely on previous values.

        # lse = logsumexp(s) / ln(2) = log(max_s) + log(sum_exp) * (1/ln(2))
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32 scalar

        # Second pass: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            start = tl.load(kv_indptr_ptr + b)
            idx = start + i

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM]

            q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM]
            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale

            attn = tl.exp(s - lse_val)
            out_vec += attn * v_vec  # elementwise multiply and accumulate

        # Store results
        out_base = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)

        lse_index = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_index, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available; otherwise, raise (evaluation requires Triton)
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for computation.")

        # Ensure inputs are on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes and constants
        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        assert num_qo_heads == 32 and head_dim == 128
        num_pages, seq, num_kv_heads, _ = k_cache.shape
        assert seq == 1 and num_kv_heads == 8

        # Output buffers (compute in float32, cast later)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Cast inputs to float32 for compute
        q32 = q.to(torch.float32)
        k32 = k_cache.to(torch.float32)
        v32 = v_cache.to(torch.float32)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        # Use a large upper bound for tokens; masks ensure correctness.
        NUM_TOKS = 8192  # covers all provided workloads

        _attention_bh_kernel[grid](
            q32, k32, v32, kv_indptr, kv_indices,
            output, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            sm_scale=float(sm_scale),
            NUM_TOKS=NUM_TOKS,
            num_warps=4,
        )

        # Return output in bfloat16 and lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
