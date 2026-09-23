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
        q_ptr,             # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        k_ptr,             # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,             # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,     # *int32, [BATCH_SIZE + 1]
        kv_indices_ptr,    # *int32, [NUM_KV_INDICES]
        out_ptr,           # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,           # *float32, [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,          # float32 scalar
        half_ln2_inv,      # float32 scalar = 1 / ln(2)
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,   # 32
        NUM_KV_HEADS: tl.constexpr,   # 8
        HEAD_DIM: tl.constexpr,       # 128
        NUM_TOKS: tl.constexpr,       # e.g., 1024
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping: 32 query heads -> 8 KV heads
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[b, h, :] vector
        q_off = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # First pass: compute logsumexp(s) where s = (q·k_i) * sm_scale
        max_s = -float("inf")
        sum_exp = 0.0

        start = tl.load(kv_indptr_ptr + b)  # int32
        end = tl.load(kv_indptr_ptr + b + 1)  # int32
        num_tokens = end - start  # int32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows corresponding to this token index
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Compute dot product q·k_i
            dot = tl.sum(q_vec * k_vec, axis=0)  # scalar float32

            s = dot * sm_scale
            if mask_i:
                max_s = tl.maximum(max_s, s)
                # sum_exp = sum_exp * exp(max_s - s) + 1 is numerically stable
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv

        # Second pass: recompute s, compute attn, accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            dot = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = dot * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32

            if mask_i:
                out_vec += attn * v_vec

        # Store outputs
        out_base = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA and contiguous; cast to float32 for compute
        device = q.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors for Triton"
        assert TRITON_AVAILABLE, "Triton is not available"

        # Shapes
        B, H_q, D = q.shape
        _, _, K_h, Dk = k_cache.shape
        assert H_q == 32 and D == 128 and K_h == 8, "Expected q: [B, 32, 128], k/v: [P, 1, 8, 128]"

        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k_cache.contiguous().to(torch.float32)
        v_f32 = v_cache.contiguous().to(torch.float32)
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)

        # Output and lse tensors in fp32 for computation
        output = torch.empty((B, H_q, D), dtype=torch.float32, device=device)  # compute in fp32
        lse = torch.empty((B, H_q), dtype=torch.float32, device=device)

        grid = (B, H_q)
        NUM_TOKS = 1024
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q_f32, k_f32, v_f32, kv_indptr_i32, kv_indices_i32,
            output, lse,
            sm_scale, half_ln2_inv,
            BATCH_SIZE=B,
            NUM_QO_HEADS=H_q,
            NUM_KV_HEADS=K_h,
            HEAD_DIM=D,
            NUM_TOKS=NUM_TOKS,
            num_warps=4,
        )

        # Return outputs as bfloat16 to match original; lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
