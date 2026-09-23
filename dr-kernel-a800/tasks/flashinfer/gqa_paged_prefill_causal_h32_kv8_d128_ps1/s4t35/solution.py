import torch
import math
import triton
import triton.language as tl


@triton.jit
def gqa_batch_kernel(
    q_ptr,          # *f32, [total_q, NUM_QO_HEADS, HEAD_DIM] contiguous
    k_ptr,          # *f32, [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM] contiguous
    v_ptr,          # *f32, [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM] contiguous
    out_ptr,        # *f32, [total_q, NUM_QO_HEADS, HEAD_DIM] contiguous
    lse_ptr,        # *f32, [len_indptr - 1, NUM_QO_HEADS] contiguous
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    kv_indices_ptr, # *i32, [num_kv_tokens] for this batch
    SM_SCALE: tl.constexpr,           # e.g., 1.0 / sqrt(128)
    NUM_QO_HEADS: tl.constexpr,       # 32
    NUM_KV_HEADS: tl.constexpr,       # 8
    HEAD_DIM: tl.constexpr,           # 128
    GQA_RATIO: tl.constexpr,          # 4
    TOTAL_Q: tl.int32,                # runtime total_q
    NUM_Q_TOKENS: tl.int32,           # runtime num_q_tokens for this batch
    NUM_KV_TOKENS: tl.int32,          # runtime num_kv_tokens for this batch
    Q_START: tl.int32,                # runtime q_start for this batch
    KV_START: tl.int32,               # runtime kv_start for this batch
):
    b = tl.program_id(0)

    # Load batch start/end
    qo_start = tl.load(qo_indptr_ptr + b)       # i32
    qo_end = tl.load(qo_indptr_ptr + b + 1)     # i32
    kv_start_b = tl.load(kv_indptr_ptr + b)     # i32
    kv_end_b = tl.load(kv_indptr_ptr + b + 1)   # i32

    # If nothing to do, return
    if qo_start >= qo_end or kv_start_b >= kv_end_b:
        return

    # Pass 1: compute lse_max and lse_sum per head
    for h in range(NUM_QO_HEADS):
        kv_head = h // GQA_RATIO
        max_logits = -float("inf")
        sum_exp = 0.0

        for i in range(NUM_KV_TOKENS):
            kv_index = tl.load(kv_indices_ptr + i)  # int32
            # Base pointers for k and v at (kv_index, kv_head)
            base_k = kv_index * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            base_v = kv_index * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            # Load k vector [HEAD_DIM]
            k_vec = tl.load(k_ptr + base_k + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
            # Load q vector for head h at arbitrary token t; we'll compute dot for each token later
            # We need q[t, h], but we'll loop over t here as well. To avoid loading q_vec per t,
            # we'll load q_vec for each t inside the second pass. For now, keep scalars.

            # We will compute dot per t in second pass; here track max and sum only based on q_vec loaded for each t.
            # To do that, we need q_vec per t. We'll skip computing max/sum without q; instead, we'll recompute logits_scaled in second pass and use them to update max/sum.

            # Note: Triton does not support Python break; we must use scalar loops.
            # We'll instead store max and sum in host if needed. For now, recompute in second pass with q.

        # Second pass: compute output and recompute lse_max/sum using actual q[t, h]
        for t in range(TOTAL_Q):
            # Only process tokens in [qo_start, qo_end)
            if t < Q_START or t >= Q_START + NUM_Q_TOKENS:
                continue

            # Compute q vector for head h at token t
            q_vec = tl.load(q_ptr + t * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM + tl.arange(0, HEAD_DIM), mask=True, other=0.0)

            # Initialize output vector for this head
            out_vec = tl.load(out_ptr + t * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM + tl.arange(0, HEAD_DIM), mask=True, other=0.0)

            # Recompute max and sum for this t and head
            max_logits = -float("inf")
            sum_exp = 0.0
            for i in range(NUM_KV_TOKENS):
                kv_index = tl.load(kv_indices_ptr + i)
                base_k = kv_index * NUM_KV_HEADS * HEAD_DIM + (h // GQA_RATIO) * HEAD_DIM
                k_vec = tl.load(k_ptr + base_k + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                dot = tl.sum(q_vec * k_vec, axis=0)
                logits_scaled = dot * SM_SCALE
                max_logits = tl.maximum(max_logits, logits_scaled)
                exp_i = tl.exp(logits_scaled - max_logits)
                sum_exp += exp_i

            # Now write lse for this b, h
            tl.store(lse_ptr + b * NUM_QO_HEADS + h, max_logits + tl.log(sum_exp))

            # Second phase: compute softmax and accumulate output
            for i in range(NUM_KV_TOKENS):
                kv_index = tl.load(kv_indices_ptr + i)
                kv_head = h // GQA_RATIO
                base_k = kv_index * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
                k_vec = tl.load(k_ptr + base_k + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                q_vec = tl.load(q_ptr + t * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                dot = tl.sum(q_vec * k_vec, axis=0)
                logits_scaled = dot * SM_SCALE
                softmax = tl.exp(logits_scaled - max_logits) / sum_exp

                base_v = kv_index * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
                v_vec = tl.load(v_ptr + base_v + tl.arange(0, HEAD_DIM), mask=True, other=0.0)

                out_vec += softmax * v_vec
            tl.store(out_ptr + t * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert int(qo_indptr[-1].item()) == total_q

        # Flatten cached k/v to [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        # Output buffers in float32 for accumulation; we will cast to bfloat16 after kernel
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((len_indptr - 1, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per batch b
        grid = (len_indptr - 1,)
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Gather q slice in float32 and contiguous
            q_batch = q[q_start:q_end].to(torch.float32).contiguous()  # [num_q_tokens, 32, 128]
            # Gather k_batch and v_batch using kv_indices slice for this batch (data movement only)
            kv_indices_batch = kv_indices[kv_start:kv_end].to(torch.int32)  # [num_kv_tokens]
            k_batch = k_cache_flat.index_select(0, kv_indices_batch).contiguous()  # [num_kv_tokens, 8, 128]
            v_batch = v_cache_flat.index_select(0, kv_indices_batch).contiguous()  # [num_kv_tokens, 8, 128]

            # Launch Triton kernel
            gqa_batch_kernel[grid](
                q_batch, k_batch, v_batch,
                output, lse,
                qo_indptr, kv_indptr, kv_indices_batch,
                SM_SCALE=float(sm_scale),
                NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
                TOTAL_Q=total_q,
                NUM_Q_TOKENS=num_q_tokens,
                NUM_KV_TOKENS=num_kv_tokens,
                Q_START=q_start,
                KV_START=kv_start,
                num_warps=1, num_stages=2
            )

        # Return output in bfloat16 as per original
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
