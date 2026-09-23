import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). It iterates up to NUM_TOKS with masks,
# computing lse and output vector for that (b, h).
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,   # 8
        HEAD_DIM: tl.constexpr,       # 128
        NUM_TOKS: tl.constexpr,       # upper bound for loop, e.g., 8192
        sm_scale: tl.float32,         # 1.0 / sqrt(128)
        half_ln2_inv: tl.float32,     # 1 / ln(2) = ~1.4426950408889634
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # scalar int32

        # Base offset for q[h]
        q_off = h * HEAD_DIM

        # Batch range from kv_indptr
        start = tl.load(kv_indptr_ptr + b)       # int32
        end = tl.load(kv_indptr_ptr + b + 1)     # int32
        num_tokens_actual = end - start          # int32 scalar

        # Accumulators for lse
        max_s = -float("inf")                    # scalar float32
        sum_exp = 0.0                            # scalar float32

        # Pass 1: compute max_s and sum_exp over tokens
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual      # scalar bool
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32
            # Offsets for k and v rows: idx * (8 * 128) + kv_head * 128
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM] float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM] float32

            # Load q[h]
            q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM] float32

            # Compute dot product
            dot = 0.0
            for d in range(HEAD_DIM):
                dot += q_vec[d] * k_vec[d]
            s = dot * sm_scale  # scalar float32

            # Update max_s and sum_exp if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # scalar float32

        # Initialize output vector
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Pass 2: recompute s, compute attn, and accumulate output
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM] float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM] float32
            q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM] float32

            dot = 0.0
            for d in range(HEAD_DIM):
                dot += q_vec[d] * k_vec[d]
            s = dot * sm_scale  # scalar float32

            attn = tl.exp(s - lse_val)  # scalar float32

            # Accumulate output vector with mask
            if mask_i:
                for d in range(HEAD_DIM):
                    out_vec[d] += attn * v_vec[d]

        # Store output and lse
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution: ensure Triton is available
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required but not available.")

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        num_kv_heads = 8
        assert num_qo_heads == 32 and head_dim == 128, "This Triton kernel assumes 32 heads and 128 dims."

        # Ensure inputs are on CUDA and contiguous; cast to float32 for compute
        device = q.device
        q32 = q.contiguous().to(torch.float32)        # [B, 32, 128]
        k32 = k_cache.contiguous().squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v32 = v_cache.contiguous().squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        # Output and lse buffers (float32 for compute)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        NUM_TOKS = 8192  # conservative upper bound; masks guard correctness

        # sm_scale and half_ln2_inv constants
        sm_scale_val = float(sm_scale)  # 1.0 / sqrt(128)
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k32, v32,
            kv_indptr, kv_indices,
            output, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            sm_scale=sm_scale_val,
            half_ln2_inv=half_ln2_inv,
            num_warps=4,
        )

        # Cast output to bfloat16 to match original output dtype; lse remains float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
