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
        B: tl.constexpr,                      # batch size
        NUM_QO_HEADS: tl.constexpr,          # 32
        NUM_KV_HEADS: tl.constexpr,          # 8
        HEAD_DIM: tl.constexpr,              # 128
        NUM_TOKS: tl.constexpr,              # max tokens per batch, e.g., 8192
        sm_scale: tl.constexpr,              # float32 scalar
    ):
        # Program id: one per (b, h)
        pid = tl.program_id(axis=0)
        b = pid // NUM_QO_HEADS
        h = pid % NUM_QO_HEADS

        # Base offsets
        q_off = h * HEAD_DIM  # q_ptr layout: [NUM_QO_HEADS, HEAD_DIM] contiguous
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # GQA mapping
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS  # 4
        kv_head = h // kv_ratio  # 0..7

        # Start/end of valid tokens for this batch
        start = tl.load(kv_indptr_ptr + b)          # int32
        end = tl.load(kv_indptr_ptr + b + 1)        # int32
        num_tokens_actual = end - start             # int32

        # Pass 1: accumulate max_s and sum_exp = sum(exp(s - max_s)) across tokens
        max_s = -1e20  # float32
        sum_exp = 0.0  # float32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked), [HEAD_DIM]
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                # sum_exp = sum_exp * exp(max_s - s) + 1
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0
            # else do nothing

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        # 1/ln(2) = 1.4426950408889634
        half_ln2_inv = 1.4426950408889634  # float32 constant
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Write lse for this (b, h)
        lse_idx = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_idx, lse_val)

        # Pass 2: recompute s, compute attn = exp(s - lse), accumulate out_vec += attn * v
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)

            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32

            if mask_i:
                out_vec += attn * v_vec  # vector add

        # Store output[b, h, :]
        out_base = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size, num_qo_heads=32, num_kv_heads=8, head_dim=128, sm_scale=None, num_toks_max=8192):
        super().__init__()
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
        self.num_toks_max = num_toks_max

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale=None):
        # Ensure Triton availability
        if not TRITON_AVAILABLE:
            # Fallback: if Triton not available, return None to indicate failure
            return None, None

        # Ensure CUDA tensors and contiguity
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Cast to float32 for compute
        q32 = q.float()
        k32 = k_cache.float()
        v32 = v_cache.float()

        B = q.shape[0]
        NUM_QO_HEADS = 32
        NUM_KV_HEADS = 8
        HEAD_DIM = 128

        # Allocate outputs
        out = torch.empty((B, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=device)  # compute in float32
        lse = torch.empty((B, NUM_QO_HEADS), dtype=torch.float32, device=device)

        # Effective sm_scale: if None, use module's default; else use input
        sm_scale_val = sm_scale if sm_scale is not None else self.sm_scale

        # Launch one program per (b, h)
        grid = (B * NUM_QO_HEADS,)
        _attention_bh_kernel[grid](
            q32, k32, v32, kv_indptr, kv_indices, out, lse,
            B=B, NUM_QO_HEADS=NUM_QO_HEADS, NUM_KV_HEADS=NUM_KV_HEADS, HEAD_DIM=HEAD_DIM,
            NUM_TOKS=self.num_toks_max, sm_scale=sm_scale_val,
            num_warps=4, num_stages=1,
        )

        # Cast output to bfloat16 to match original, return as tuple
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
