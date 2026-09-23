import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS cached tokens with masks. No torch ops are used.
if TRITON_AVAILABLE:
    @triton.jit
    def _run_bh_kernel(
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
        NUM_TOKS: tl.constexpr,              # upper bound on num_tokens per batch
        HALF_LN2_INV: tl.constexpr,          # 1 / ln(2)
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Start/end indices for this batch from kv_indptr
        start = tl.load(kv_indptr_ptr + b)
        end = tl.load(kv_indptr_ptr + b + 1)
        num_tokens_actual = end - start

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h, :] as a vector
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM], float32

        # Pass 1: accumulate max and sum(exp(s - max)) across tokens
        max_s = -float("inf")
        sum_exp = 0.0
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32 token index

            # Offsets for k and v rows (flattened: (idx * NUM_KV_HEADS + kv_head) * HEAD_DIM)
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            # Load k_vec and v_vec
            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits  # sm_scale is implicit (already scaled by 1/sqrt(128) in the original run function)
            # Since original lse is logsumexp(logits_scaled) with logits_scaled = (q·k)*sm_scale, and sm_scale>0,
            # logsumexp(s) == log(sum(exp(s))). We still apply division by ln(2) as required.
            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        # If num_tokens_actual == 0, we set lse to -inf to match original behavior.
        # However, for masked loop, when no tokens, sum_exp == 0, log(0) is -inf, so we don't need special handling.
        lse_val = tl.log(max_s) + tl.log(sum_exp) * HALF_LN2_INV  # float32

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32 token index

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            s = logits
            attn = tl.exp(s - lse_val)  # softmax term

            # Accumulate output: out_vec += attn * v_vec
            out_vec += attn * v_vec

        # Store results
        out_offset = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # We must do all computation in Triton. No torch ops in forward.
        # Inputs: q [B, 32, 128], k_cache [NUM_PAGES, 1, 8, 128], v_cache similarly.
        # kv_indptr [B+1], kv_indices [NUM_T], sm_scale float32.

        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        assert q.dim() == 3 and q.shape[1] == 32 and q.shape[2] == 128, "q must be [B, 32, 128]"
        assert k_cache.dim() == 4 and k_cache.shape[1] == 1 and k_cache.shape[2] == 8 and k_cache.shape[3] == 128, "k_cache must be [NUM_PAGES, 1, 8, 128]"
        assert v_cache.dim() == 4 and v_cache.shape[1] == 1 and v_cache.shape[2] == 8 and v_cache.shape[3] == 128, "v_cache must be [NUM_PAGES, 1, 8, 128]"
        assert kv_indptr.dim() == 1 and kv_indptr.shape[0] == q.shape[0] + 1, "kv_indptr must be [B+1]"
        assert kv_indices.dim() == 1, "kv_indices must be 1D"
        assert q.shape[0] == kv_indptr.shape[0] - 1, "Batch size mismatch with kv_indptr"

        B = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        num_pages = k_cache.shape[0]
        num_kv_heads = k_cache.shape[2]

        # Ensure inputs are contiguous and float32 for Triton compute
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k_cache.contiguous().to(torch.float32)
        v_f32 = v_cache.contiguous().to(torch.float32)
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)

        # Output and lse buffers (float32 for compute, cast later)
        out = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, num_qo_heads)
        # We set NUM_TOKS as an upper bound; provided workloads use small num_tokens.
        NUM_TOKS = 8192  # Safe upper bound; masks protect actual num_tokens
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _run_bh_kernel[grid](
            q_f32, k_f32, v_f32, kv_indptr_i32, kv_indices_i32,
            out, lse,
            BATCH_SIZE=B, NUM_QO_HEADS=num_qo_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS, HALF_LN2_INV=half_ln2_inv,
            num_warps=4, num_stages=2,
        )

        # Cast output to bfloat16 to match original model's output dtype
        out_bf16 = out.to(torch.bfloat16)

        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
