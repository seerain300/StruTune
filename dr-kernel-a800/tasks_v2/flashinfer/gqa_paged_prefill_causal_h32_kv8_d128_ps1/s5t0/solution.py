import torch
import math
import triton
import triton.language as tl

# Constants specialized to the original code
HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # 4
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)  # 1.0 / sqrt(128)


@triton.jit
def process_one_qh(
    q_ptr,            # *fp32, [total_q, NUM_QO_HEADS, HEAD_DIM]
    k_flat_ptr,       # *fp32, [num_pages, NUM_KV_HEADS, HEAD_DIM]
    v_flat_ptr,       # *fp32, [num_pages, NUM_KV_HEADS, HEAD_DIM]
    out_ptr,          # *fp32, [total_q, NUM_QO_HEADS, HEAD_DIM]
    lse_ptr,          # *fp32, [total_q, NUM_QO_HEADS]
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    total_q,          # int32
    len_indptr,       # int32
    num_q_tokens_in_b,  # int32 (passed as size_t), number of queries in this batch element
    num_kv_indices,      # int32
    sm_scale,         # fp32 scalar
    # grid: (len_indptr - 1, num_q_tokens_in_b, NUM_QO_HEADS)
    # pid0 is the batch element index (b), pid1 is q_idx, pid2 is h
    BLOCK_H: tl.constexpr,    # usually 1
    BLOCK_D: tl.constexpr,    # usually HEAD_DIM, could be 128
):
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Compute q range
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    num_q_tokens = q_end - q_start

    # If q_idx >= num_q_tokens, early exit (shouldn't happen if grid matches, but guard anyway)
    if q_idx >= num_q_tokens:
        return

    # Compute kv range for this batch element
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_kv_indices_in_b = kv_end - kv_start

    # Global query index
    global_q_idx = q_start + q_idx

    # Causal masking: delta = num_kv_indices_in_b - num_q_tokens
    # Note: num_q_tokens is for this batch element (b), not global total_q
    delta = num_kv_indices_in_b - num_q_tokens

    # max number of kv tokens to consider for this query index
    # q_idx + 1 + delta: consider all but the last (self-attention-like) plus delta shift
    # Use Python-like min; Triton has tl.minimum
    # But we need to compute q_idx + 1 + delta in Triton
    # Triton supports arithmetic; build a vector for max_kv_idx candidate
    q_idx_plus_one = q_idx + 1
    candidate_max = q_idx_plus_one + delta
    max_kv_idx = tl.minimum(candidate_max, num_kv_indices_in_b)

    # If max_kv_idx <= 0, skip
    if max_kv_idx <= 0:
        return

    # Map query head to kv head (GQA)
    kv_head = h // GQA_RATIO  # integer division

    # Load q vector for this head: q[global_q_idx, h, :]
    # Address: q_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM + d
    q_offsets = tl.arange(0, BLOCK_D)
    q_mask = q_offsets < HEAD_DIM
    q_vec = tl.load(
        q_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM + q_offsets,
        mask=q_mask,
        other=0.0
    )  # shape [BLOCK_D], dtype fp32

    # Gather k and v sub-blocks for this batch element and kv head
    # k_flat_ptr[idx, kv_head, d] where idx in [kv_start, kv_start + max_kv_idx)
    # Address: k_flat_ptr + idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + d
    # v_flat_ptr similar
    kv_offsets = tl.arange(0, max_kv_idx)  # [max_kv_idx]
    k_ptrs = kv_indices_ptr + kv_start + kv_offsets  # indices into kv_indices
    # Load k and v indices for this b; k_ptrs now point into kv_indices
    # For each offsets, load k values
    # Build 2D pointer for k and v: [max_kv_idx, HEAD_DIM]
    # k_base[d] = k_flat_ptr + (k_idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + d)
    k_vals = tl.zeros((max_kv_idx, HEAD_DIM), dtype=tl.float32)
    v_vals = tl.zeros((max_kv_idx, HEAD_DIM), dtype=tl.float32)

    for i in range(max_kv_idx):
        k_idx = tl.load(k_ptrs + i).to(tl.int32)
        k_base = k_flat_ptr + k_idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_base = v_flat_ptr + k_idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vals[i, :] = tl.load(k_base + tl.arange(0, HEAD_DIM), mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM), other=0.0)
        v_vals[i, :] = tl.load(v_base + tl.arange(0, HEAD_DIM), mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM), other=0.0)

    # Compute logits = q_vec @ k_vals.T -> [max_kv_idx]
    # q_vec: [HEAD_DIM], k_vals.T: [HEAD_DIM, max_kv_idx]
    logits = tl.zeros((max_kv_idx,), dtype=tl.float32)
    for j in range(max_kv_idx):
        k_row = k_vals[j, :]  # [HEAD_DIM]
        logits[j] = tl.sum(q_vec * k_row, axis=0)

    # Scale
    logits_scaled = logits * sm_scale

    # logsumexp base-2 for the whole vector
    # First compute max for stability
    max_logit = tl.max(logits_scaled, axis=0)
    # exponentiate
    exp_vals = tl.exp(logits_scaled - max_logit)
    # sum
    sum_exp = tl.sum(exp_vals, axis=0)
    lse_val = max_logit + tl.log(sum_exp)  # natural log
    lse_base2 = lse_val / tl.log(2.0)  # base-2 logsumexp

    # Softmax over logits_scaled
    probs = exp_vals / sum_exp

    # Compute output head = probs @ v_vals -> [HEAD_DIM]
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for j in range(max_kv_idx):
        v_row = v_vals[j, :]  # [HEAD_DIM]
        out_vec += probs[j] * v_row

    # Store to output[global_q_idx, h, :]
    out_base = out_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    tl.store(out_base + tl.arange(0, HEAD_DIM), out_vec, mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM))

    # Store lse[global_q_idx, h]
    lse_base = lse_ptr + global_q_idx * NUM_QO_HEADS + h
    tl.store(lse_base, lse_base2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        assert q.is_cuda, "Triton requires CUDA tensors. Please move inputs to CUDA."
        assert q.dtype == torch.bfloat16, "q must be bfloat16"
        q = q.contiguous()
        # Flatten k/v cache since original asserts and uses squeeze(1)
        k_flat = k_cache.squeeze(1).contiguous().to(torch.float32)
        v_flat = v_cache.squeeze(1).contiguous().to(torch.float32)
        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        total_q = q.shape[0]
        len_indptr = qo_indptr.shape[0]
        # num_q_tokens_in_b is per-batch element; we need to pass this to Triton
        # We'll compute grid per b by launching a loop over b. Triton supports passing scalars.
        # But Triton grid is static; we'll compute b's num_q_tokens on host and pass as meta?
        # Triton expects grid; we can compute num_q_tokens_in_b for each b and launch per b.
        # However, Triton kernel launch expects grid known at launch. We can do:
        # 1) Run loop over b in host, compute num_q_tokens_in_b, and launch kernel per b
        # 2) Compute num_q_tokens_in_b outside and pass as argument.
        # We choose option 1: launch per b, compute num_q_tokens_in_b, and set grid accordingly.

        # Allocate outputs in fp32 for numerical stability
        output_fp32 = torch.empty((total_q, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse_fp32 = torch.empty((total_q, NUM_QO_HEADS), dtype=torch.float32, device=q.device)

        # We'll iterate b on host to determine grid (bloop), since Triton grid must be static
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            num_q_tokens_in_b = q_end - q_start
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_indices_in_b = kv_end - kv_start

            # Launch Triton kernel for this batch element. Grid is (1, num_q_tokens_in_b, NUM_QO_HEADS).
            grid = (1, num_q_tokens_in_b, NUM_QO_HEADS)

            process_one_qh[grid](
                q_ptr=q, k_flat_ptr=k_flat, v_flat_ptr=v_flat,
                out_ptr=output_fp32, lse_ptr=lse_fp32,
                qo_indptr_ptr=qo_indptr, kv_indptr_ptr=kv_indptr, kv_indices_ptr=kv_indices,
                total_q=total_q, len_indptr=len_indptr,
                num_q_tokens_in_b=num_q_tokens_in_b,
                num_kv_indices=int(num_kv_indices_in_b),
                sm_scale=sm_scale,
                BLOCK_H=1, BLOCK_D=HEAD_DIM,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 to match original return type
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse_fp32

# The following helper functions and original run are kept as-is for testing convenience.
# You can call ModelNew().forward(*get_inputs()) to compare with the original run(*get_inputs()).


def run(*args):
    return ModelNew()(*args)
