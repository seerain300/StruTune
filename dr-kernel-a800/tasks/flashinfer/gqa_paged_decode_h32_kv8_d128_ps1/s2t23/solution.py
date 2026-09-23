import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It iterates up to NUM_TOKS with masks and computes:
# - lse[b, h] = logsumexp((q[b,h]·k_i) * sm_scale) / ln(2)
# - output[b, h, :] = sum_i softmax((q[b,h]·k_i) * sm_scale) * v_i
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_triton(
        q_ptr,           # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        k_ptr,           # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,           # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,   # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,  # *int32, shape [NUM_KV_INDICES]
        out_ptr,         # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,         # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SM_SCALE: tl.float32,
        HALF_LN2_INV: tl.float32,
        NUM_TOKS: tl.constexpr,
        sm_scale: tl.float32,
        half_ln2_inv: tl.float32,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Compute GQA kv head mapping
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Base offset for q[b, h, :]
        q_off = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Running max and sum for logsumexp(s) where s = (q·k_i) * sm_scale
        max_s = -float("inf")
        sum_exp = 0.0

        # Pass 1: compute max and sum_exp across tokens
        for i in range(NUM_TOKS):
            mask_i = i < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
            # token index for this iteration
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b] + i, mask=mask_i, other=0)  # int32
            # Offsets for k and v rows at kv_head
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product for this token
            dot_i = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s_i = dot_i * sm_scale

            # Update max and sum_exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s_i)
                # sum_exp = sum_exp * exp(max_s - s_i) + 1
                sum_exp = sum_exp * tl.exp(max_s - s_i) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv
        # Store lse to lse[b, h]
        lse_off = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_off, lse_val)

        # Pass 2: recompute s_i, compute attn_i = exp(s_i - lse), accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b] + i, mask=mask_i, other=0)  # int32

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            dot_i = tl.sum(q_vec * k_vec, axis=0)
            s_i = dot_i * sm_scale
            attn_i = tl.exp(s_i - lse_val)
            # Accumulate output
            out_vec += attn_i * v_vec

        # Store output[b, h, :]
        out_off = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        # Scaling constant
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        # 1 / ln(2)
        self.half_ln2_inv = 1.4426950408889634  # math.log(2.0) ** -1

        # Triton meta-params
        self.num_toks = 1024  # upper bound for token iterations; masks guard out-of-range

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices):
        # Ensure CUDA and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        assert num_qo_heads == self.num_qo_heads and head_dim == self.head_dim

        # Cast inputs to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k_cache.to(torch.float32)
        v_f32 = v_cache.to(torch.float32)

        # Allocate outputs (float32 for compute; will cast to bfloat16 before return)
        out = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        # Note: Triton requires passing sm_scale and half_ln2_inv as kwargs
        _attention_bh_triton[grid](
            q_f32, k_f32, v_f32, kv_indptr, kv_indices, out, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=self.num_kv_heads,  # 8
            HEAD_DIM=head_dim,
            SM_SCALE=self.sm_scale,
            HALF_LN2_INV=self.half_ln2_inv,
            NUM_TOKS=self.num_toks,
            sm_scale=self.sm_scale,
            half_ln2_inv=self.half_ln2_inv,
            num_warps=4,  # reasonable default for small vectors
            num_stages=2,
        )

        # Return output cast to bfloat16 and lse as float32
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
