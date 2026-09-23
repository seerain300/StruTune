import torch
import math
import triton
import triton.language as tl


@triton.jit
def run_kernel(
    q_ptr,          # *bfloat16, [total_q, NUM_QO_HEADS, HEAD_DIM], flattened
    k_ptr,          # *float32, [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM], flattened
    v_ptr,          # *float32, [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM], flattened
    out_ptr,        # *bfloat16, [total_q, NUM_QO_HEADS, HEAD_DIM], flattened
    lse_ptr,        # *float32, [len_indptr - 1, NUM_QO_HEADS]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    kv_indices_ptr, # *i32, [num_kv_indices]
    NUM_QO_HEADS: tl.constexpr,     # 32
    NUM_KV_HEADS: tl.constexpr,     # 8
    HEAD_DIM: tl.constexpr,         # 128
    GQA_RATIO: tl.constexpr,        # 4
    SM_SCALE: tl.constexpr,         # 1.0 / sqrt(128)
):
    # program ids: batch b and global token index within all batches
    b = tl.program_id(0)
    token_global = tl.program_id(1)

    q_start = tl.load(qo_indptr_ptr + b)       # i32
    q_end = tl.load(qo_indptr_ptr + b + 1)     # i32
    kv_start = tl.load(kv_indptr_ptr + b)      # i32
    kv_end = tl.load(kv_indptr_ptr + b + 1)    # i32

    # If this batch has no queries or kv, skip
    if q_start >= q_end or kv_start >= kv_end:
        return

    # Map global token index to local index in this batch
    local_q_idx = token_global - q_start
    if local_q_idx < 0 or local_q_idx >= (q_end - q_start):
        return

    # For each query head h
    for h in range(NUM_QO_HEADS):
        kv_head = h // GQA_RATIO  # GQA mapping: 32 heads -> 8 kv heads

        # Load q_vec[h] for this token (cast to float32 for compute)
        # q layout: [total_q, NUM_QO_HEADS, HEAD_DIM] flattened
        q_row_stride = NUM_QO_HEADS * HEAD_DIM
        q_row_base = q_start * q_row_stride + h * HEAD_DIM  # local base for this batch
        q_vec = tl.load(q_ptr + q_row_base + local_q_idx * q_row_stride + h * HEAD_DIM + tl.arange(0, HEAD_DIM),
                        mask=tl.arange(0, HEAD_DIM) < HEAD_DIM,
                        other=0.0).to(tl.float32)

        # Initialize accumulators
        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        # num_kv_tokens for this batch
        num_kv_tokens = kv_end - kv_start  # scalar int32

        # Loop over KV tokens and compute attention
        for i in range(num_kv_tokens):
            kv_index = tl.load(kv_indices_ptr + kv_start + i).to(tl.int32)

            # k/v layout: [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM] flattened
            kv_row_stride = NUM_KV_HEADS * HEAD_DIM
            k_vec = tl.load(k_ptr + (kv_index * kv_row_stride) + kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM),
                            mask=tl.arange(0, HEAD_DIM) < HEAD_DIM,
                            other=0.0).to(tl.float32)
            logits = tl.sum(q_vec * k_vec, axis=0) * SM_SCALE  # scalar
            new_max = tl.maximum(running_max, logits)
            p = tl.exp(logits - new_max)
            sum_p = running_sum * tl.exp(running_max - new_max) + p
            running_max = new_max
            running_sum = sum_p

            v_vec = tl.load(v_ptr + (kv_index * kv_row_stride) + kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM),
                            mask=tl.arange(0, HEAD_DIM) < HEAD_DIM,
                            other=0.0).to(tl.float32)
            acc += p * v_vec

        # Final lse and output store
        lse_val = tl.log(running_sum) + running_max  # scalar float32
        # Store acc as bfloat16 into output[q_start + local_q_idx, h, :]
        out_row_base = (q_start + local_q_idx) * q_row_stride + h * HEAD_DIM
        tl.store(out_ptr + out_row_base + tl.arange(0, HEAD_DIM), acc.to(tl.bfloat16), mask=tl.arange(0, HEAD_DIM) < HEAD_DIM)
        # Store lse for this (batch, head) into lse[b, h]
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)


def _run_triton(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    device = q.device
    total_q = q.shape[0]
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    gqa_ratio = 4  # 32 // 8
    sm_scale = float(sm_scale)

    # Flatten cache to [num_pages, num_kv_heads, head_dim] and keep dtype float32
    k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()
    v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

    # Precompute per-batch q_start/q_end and kv_start/kv_end
    len_indptr = qo_indptr.shape[0]
    q_batches = [(int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())) for i in range(len_indptr - 1)]
    kv_batches = [(int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())) for i in range(len_indptr - 1)]

    # Gather k/v per batch using torch.index_select to handle dynamic kv_indices
    k_b = []
    v_b = []
    for b in range(len_indptr - 1):
        start, end = kv_batches[b]
        if start >= end:
            k_b.append(torch.empty((0, num_kv_heads, head_dim), dtype=torch.float32, device=device))
            v_b.append(torch.empty((0, num_kv_heads, head_dim), dtype=torch.float32, device=device))
        else:
            kv_indices_batch = kv_indices[start:end].to(torch.int64)
            k_rows = k_cache_flat.index_select(0, kv_indices_batch).contiguous()
            v_rows = v_cache_flat.index_select(0, kv_indices_batch).contiguous()
            k_b.append(k_rows)
            v_b.append(v_rows)

    # Concatenate per-batch k/v into total_k and total_v
    total_k = torch.cat([k_b[i] for i in range(len(kv_batches))], dim=0) if len(kv_batches) > 0 else torch.empty((0, num_kv_heads, head_dim), dtype=torch.float32, device=device)
    total_v = torch.cat([v_b[i] for i in range(len(kv_batches))], dim=0) if len(kv_batches) > 0 else torch.empty((0, num_kv_heads, head_dim), dtype=torch.float32, device=device)
    num_kv_total = total_k.shape[0] if total_k.numel() > 0 else 0

    # Output buffers: lse [len_indptr-1, 32] float32; output [total_q, 32, 128] bfloat16
    lse = torch.empty((len_indptr - 1, num_qo_heads), dtype=torch.float32, device=device)
    output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)

    # Launch Triton kernel over grid (len_indptr-1, total_q)
    grid = (len_indptr - 1, total_q)
    run_kernel[grid](
        q, total_k, total_v, output, lse, qo_indptr, kv_indptr, kv_indices,
        NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4, SM_SCALE=sm_scale,
        num_warps=4, num_stages=2
    )

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward
        return _run_triton(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
