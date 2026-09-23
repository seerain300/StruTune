import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS tokens with masks and performs:
# - First pass: compute lse = logsumexp(scaled_dot) / ln(2) over valid tokens.
# - Second pass: recompute s, compute attn = exp(s - lse), and accumulate output vector.
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM] but we index by token idx
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM] but we index by token idx
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,       # float32 scalar
        NUM_TOKS: tl.constexpr,       # upper bound for loop
    ):
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # Base offset for q vector for this head
        q_off = h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Initialize lse accumulators
        max_s = -float("inf")
        sum_exp = 0.0

        # Pass 1: compute max_s and sum_exp across tokens
        for i in range(NUM_TOKS):
            # Mask: only valid i contributes
            valid = i < (tl.load(kv_indptr_ptr + b + 1, eviction_policy='evict_last') - tl.load(kv_indptr_ptr + b, eviction_policy='evict_last'))
            # Note: Triton scalar control is not allowed; rely on mask in arithmetic. Use other=0 for loads if needed.
            # We'll reconstruct 'start' from b (it's kv_indptr[b]), but Triton scalar load of b's indptr:
            start = tl.load(kv_indptr_ptr + b)  # int32
            end = tl.load(kv_indptr_ptr + b + 1)  # int32
            valid = i < (end - start)

            if valid:
                idx = tl.load(kv_indices_ptr + start + i)  # int32 token index
                kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
                kv_head = h // kv_ratio  # 0..7

                # Offsets for k and v rows (for kv_head and idx)
                k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
                v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

                # Load k_vec and v_vec (masked: if valid, always load; valid is just a scalar)
                k_vec = tl.load(k_ptr + k_off)  # [HEAD_DIM], float32
                v_vec = tl.load(v_ptr + v_off)  # [HEAD_DIM], float32

                # Dot product: scalar logits
                logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
                s = logits * sm_scale  # scaled

                # Update max and sum-exp
                max_s = tl.maximum(max_s, s)
                # sum_exp = sum_exp * exp(max_s - s) + 1
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # scalar float32

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            start = tl.load(kv_indptr_ptr + b)
            end = tl.load(kv_indptr_ptr + b + 1)
            valid = i < (end - start)

            if valid:
                idx = tl.load(kv_indices_ptr + start + i)  # int32 token index
                kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
                kv_head = h // kv_ratio

                k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
                v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

                k_vec = tl.load(k_ptr + k_off)  # [HEAD_DIM], float32
                v_vec = tl.load(v_ptr + v_off)  # [HEAD_DIM], float32

                logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
                s = logits * sm_scale
                attn = tl.exp(s - lse_val)  # float32 scalar

                # Accumulate output vector
                out_vec += attn * v_vec

        # Store results
        out_base = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        device = q.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Extract shapes
        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, seq, num_kv_heads, _ = k_cache.shape
        assert seq == 1 and num_kv_heads == 8 and head_dim == 128

        # Flags for batches with tokens
        has_tokens = (kv_indptr[1:] - kv_indptr[:-1]).clamp(min=0).to(torch.int32) > 0  # [batch_size] int32 (0/1)

        # Output buffers (compute in float32, cast later)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Initialize outputs for batches without tokens to zeros
        if has_tokens.any():
            # Cast inputs to float32 for compute
            q32 = q.to(torch.float32)
            k32 = k_cache.to(torch.float32)
            v32 = v_cache.to(torch.float32)

            # Launch Triton kernel: one program per (b, h)
            grid = (batch_size, num_qo_heads)
            NUM_TOKS = 8192  # upper bound; masks guard correctness

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
        else:
            # If no tokens for any batch, output zeros and lse = -inf
            output.zero_()
            lse.zero_()
            lse.fill_(-float("inf"))

        # Return output in bfloat16 and lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
