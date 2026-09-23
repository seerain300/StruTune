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
        BATCH_SIZE: tl.constexpr,          # meta-args for math
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,            # fixed upper bound for loop
        sm_scale,                            # float32 scalar
    ):
        # Each program handles one (b, h)
        pid = tl.program_id(0)
        b = pid // NUM_QO_HEADS
        h = pid % NUM_QO_HEADS

        # Load q[h]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Determine number of tokens for this batch (runtime scalar int32)
        start = tl.load(kv_indptr_ptr + b)            # int32
        end = tl.load(kv_indptr_ptr + b + 1)         # int32
        num_tokens_actual = end - start              # int32 scalar

        # Pass 1: compute max_s and sum_exp = sum(exp(s - max_s))
        max_s = -float("inf")
        sum_exp = 0.0

        # GQA mapping: kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

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
            sum_exp = tl.where(mask_i, sum_exp * tl.exp(max_s - s) + 1.0, sum_exp)
            max_s = tl.where(mask_i, tl.maximum(max_s, s), max_s)

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32

            # Accumulate: out_vec += attn * v_vec (masked)
            out_vec += tl.where(mask_i, attn * v_vec, v_vec * 0.0)

        # Store outputs
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for this workload
        self.NUM_QO_HEADS = 32
        self.NUM_KV_HEADS = 8
        self.HEAD_DIM = 128
        # Fixed upper bound for token loop (mask protects correctness)
        self.NUM_TOKS = 8192  # covers all provided workloads safely

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous, cast to float32 for compute
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be CUDA tensors"
        device = q.device

        B = q.shape[0]
        NUM_QO_HEADS = self.NUM_QO_HEADS
        HEAD_DIM = self.HEAD_DIM

        # Cast inputs to float32 for numerical stability; keep original device
        q_f32 = q.to(torch.float32).contiguous()                  # [B, 32, 128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()     # [N, 1, 8, 128]
        v_cache_f32 = v_cache.to(torch.float32).contiguous()     # [N, 1, 8, 128]
        kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()   # [B+1]
        kv_indices_i32 = kv_indices.to(torch.int32).contiguous() # [M]

        # Allocate outputs (compute in float32; cast later to bfloat16)
        output = torch.empty((B, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
        lse = torch.empty((B, NUM_QO_HEADS), dtype=torch.float32, device=device)

        # Flatten q to [NUM_QO_HEADS, HEAD_DIM] for kernel access
        q_vec = q_f32.reshape(NUM_QO_HEADS, HEAD_DIM).contiguous()

        # Prepare kernel launch: one program per (b, h)
        grid = (B * NUM_QO_HEADS,)

        # Launch Triton kernel (no torch ops in forward)
        _attention_bh_kernel[grid](
            q_vec,
            k_cache_f32,
            v_cache_f32,
            kv_indptr_i32,
            kv_indices_i32,
            output,
            lse,
            BATCH_SIZE=B,
            NUM_QO_HEADS=NUM_QO_HEADS,
            NUM_KV_HEADS=self.NUM_KV_HEADS,
            HEAD_DIM=HEAD_DIM,
            NUM_TOKS=self.NUM_TOKS,
            sm_scale=float(sm_scale),
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original expected dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
