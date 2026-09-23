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
        q_ptr,                 # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
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
        NUM_TOKS: tl.constexpr,
        HALF_LN2_INV: tl.constexpr,  # 1 / ln(2)
    ):
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Determine token range for this batch
        start = tl.load(kv_indptr_ptr + b)              # int32
        end = tl.load(kv_indptr_ptr + b + 1)           # int32
        num_tokens_actual = end - start                # int32

        # Load q[b, h] vector
        q_vec = tl.load(q_ptr + b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

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
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32 token index

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)              # scalar float32
            if mask_i:
                out_vec += attn * v_vec

        # Store out[b, h, :]
        out_ptr_base = out_ptr + b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size=None, num_qo_heads=32, num_kv_heads=8, head_dim=128, num_tokens_upper=8192):
        super().__init__()
        # We keep constants as instance attributes to match the original signature and axes
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_tokens_upper = num_tokens_upper

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        assert q.dtype == torch.bfloat16, "q must be bfloat16 (as in original)"
        # Cast to float32 for compute
        q_f32 = q.to(torch.float32)
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()

        BATCH_SIZE = q_f32.shape[0]
        NUM_QO_HEADS = self.num_qo_heads
        NUM_KV_HEADS = self.num_kv_heads
        HEAD_DIM = self.head_dim
        NUM_TOKS = self.num_tokens_upper

        # Allocate output and lse (float32 for compute)
        output = torch.empty((BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=q_f32.device)
        lse = torch.empty((BATCH_SIZE, NUM_QO_HEADS), dtype=torch.float32, device=q_f32.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (BATCH_SIZE, NUM_QO_HEADS)
        # Compute 1/ln(2)
        HALF_LN2_INV = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q_f32, k_cache_f32, v_cache_f32,
            kv_indptr, kv_indices,
            output, lse,
            sm_scale,
            BATCH_SIZE=BATCH_SIZE,
            NUM_QO_HEADS=NUM_QO_HEADS,
            NUM_KV_HEADS=NUM_KV_HEADS,
            HEAD_DIM=HEAD_DIM,
            NUM_TOKS=NUM_TOKS,
            HALF_LN2_INV=HALF_LN2_INV,
            num_warps=4,  # reasonable for small vectors
            num_stages=2,
        )

        # Cast output back to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
