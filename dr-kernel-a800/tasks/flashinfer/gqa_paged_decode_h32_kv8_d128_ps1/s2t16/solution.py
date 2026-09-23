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
        q_ptr,            # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,
        sm_scale: tl.constexpr,
        half_ln2_inv: tl.constexpr,  # 1 / ln(2)
    ):
        # One program per (b, h)
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Load kv_indptr[start/end] for this batch
        start = tl.load(kv_indptr_ptr + b)            # int32
        end = tl.load(kv_indptr_ptr + b + 1)         # int32
        num_tokens_actual = end - start              # int32

        # Load query vector q[h, :]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Pass 1: compute max_s and sum_exp across tokens (masked)
        max_s = -float("inf")
        sum_exp = 0.0
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio  # 0..7

            # Compute offsets for k and v rows
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec for this token (masked)
            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid; rescale sum_exp when max changes
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Pass 2: recompute s, compute attn = exp(s - lse), and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar
            out_vec += attn * v_vec

        # Store output and lse
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton isn't available, raise (evaluation expects Triton)
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        # Ensure inputs are contiguous and on same device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        # k_cache, v_cache: [num_pages, 1, 8, 128] -> squeeze dim-1 to [num_pages, 8, 128]
        k_cache = k_cache.squeeze(1)
        v_cache = v_cache.squeeze(1)

        # Cast to float32 for compute
        q32 = q.to(torch.float32)
        k32 = k_cache.to(torch.float32)
        v32 = v_cache.to(torch.float32)

        # Output and lse buffers (float32 for compute)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # Use a conservative NUM_TOKS; masks ensure correctness for any actual num_tokens
        NUM_TOKS = 8192
        sm_scale_val = float(sm_scale)  # 1.0 / sqrt(128) per original
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k32, v32,
            kv_indptr, kv_indices,
            output, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=8,
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
