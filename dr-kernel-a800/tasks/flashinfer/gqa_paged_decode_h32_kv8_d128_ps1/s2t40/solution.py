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
        q_ptr,               # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,               # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,               # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,       # *int32, shape [BATCH_SIZE + 1]
        kv_indices_ptr,      # *int32, shape [NUM_KV_INDICES]
        out_ptr,             # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,             # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.float32,
        half_ln2_inv: tl.float32,   # 1 / ln(2)
        NUM_TOKS: tl.constexpr,     # upper bound (masking for safety)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // NUM_QO_HEADS
        h = pid % NUM_QO_HEADS

        # Read kv_indptr to get token range for this batch
        start = tl.load(kv_indptr_ptr + b)            # int32
        end = tl.load(kv_indptr_ptr + b + 1)         # int32
        num_tokens_actual = end - start              # int32

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] as vector
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Accumulators for logsumexp of s = q·k * sm_scale
        max_s = -float("inf")
        sum_exp = 0.0

        # Pass 1: compute max_s and sum_exp across tokens (masked)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32 token index

            # Offsets in k_ptr and v_ptr: [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM]
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            s = logits * sm_scale

            # Update running max and sum with rescaling
            if mask_i:
                new_max = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - new_max) + 1.0
                max_s = new_max

        # lse = log(max_s) + log(sum_exp) * (1 / ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv

        # Pass 2: recompute s_i, attn_i, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM]

            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar
            out_vec = out_vec + (attn if mask_i else 0.0) * v_vec

        # Store output and lse
        out_index = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_index, out_vec)

        lse_index = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_index, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Require Triton to be available; otherwise, raise an error (evaluation uses Triton).
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        # Ensure on CUDA and contiguous
        device = q.device
        assert device.type == "cuda", "Inputs must be on CUDA device"

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, k_p, num_kv_heads, v_dim = k_cache.shape
        assert k_p == 1 and v_dim == head_dim, "k_cache/v_cache shape mismatch"

        # GQA ratio
        kv


def run(*args):
    return ModelNew()(*args)
