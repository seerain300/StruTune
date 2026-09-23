import torch
import math
import triton
import triton.language as tl

# Fixed constants per the original code
HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # 4
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)      # ~0.084089641


@triton.jit
def process_b_q(
    q_ptr,            # *fp32, [total_q, NUM_QO_HEADS, HEAD_DIM]
    k_flat_ptr,       # *fp32, [num_pages, NUM_KV_HEADS, HEAD_DIM]
    v_flat_ptr,       # *fp32, [num_pages, NUM_KV_HEADS, HEAD_DIM]
    out_ptr,          # *fp32, [total_q, NUM_QO_HEADS, HEAD_DIM]
    lse_ptr,          # *fp32, [total_q, NUM_QO_HEADS]
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    len_indptr,       # int32
    total_q,          # int32
    num_q_tokens_in_b,  # int32
    num_kv_indices_in_b,  # int32
    sm_scale,         # fp32
):
    # program_id(0): b in [0..len_indptr-2]; program_id(1): q_idx in [0..num_q_tokens_in_b-1]
    b = tl.program_id(0)
    q_idx = tl.program_id(1)

    # Load ranges for this batch element
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    global_q_idx = q_start + q_idx

    # Causal masking: delta = number of kv indices - number of queries in this batch
    delta = num_kv_indices_in_b - num_q_tokens_in_b
    q_idx_plus_one = q_idx + 1
    candidate_max = q_idx_plus_one + delta
    max_kv_idx = tl.minimum(candidate_max, num_kv_indices_in_b)

    # If no valid kv for this query in this batch, skip
    if max_kv_idx <= 0:
        return

    # Loop over all query heads h
    for h in range(NUM_QO_HEADS):
        kv_head = h // GQA_RATIO  # h // 4

        # Load q vector for this head
        q_base = q_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_base + tl.arange(0, HEAD_DIM), mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM), other=0.0)  # [HEAD_DIM], fp32

        # Build k and v slices: indices [kv_start, kv_start + max_kv_idx), per kv_head
        kv_offsets = tl.arange(0, max_kv_idx)  # [max_kv_idx]
        k_ptrs = kv_indices_ptr + kv_start + kv_offsets  # [max_kv_idx] indices into cached blocks
        k_vals = tl.zeros((max_kv_idx, HEAD_DIM), dtype=tl.float32)
        v_vals = tl.zeros((max_kv_idx, HEAD_DIM), dtype=tl.float32)

        for i in range(max_kv_idx):
            k_idx = tl.load(k_ptrs + i).to(tl.int32)
            k_base = k_flat_ptr + k_idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_base = v_flat_ptr + k_idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            k_vals[i, :] = tl.load(k_base + tl.arange(0, HEAD_DIM), mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM), other=0.0)
            v_vals[i, :] = tl.load(v_base + tl.arange(0, HEAD_DIM), mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM), other=0.0)

        # Compute logits = q_vec @ k_vals.T -> [max_kv_idx]
        logits = tl.zeros((max_kv_idx,), dtype=tl.float32)
        for j in range(max_kv_idx):
            k_row = k_vals[j, :]  # [HEAD_DIM]
            logits[j] = tl.sum(q_vec * k_row, axis=0)

        # Scale by sm_scale
        logits_scaled = logits * sm_scale

        # logsumexp base-2
        max_logit = tl.max(logits_scaled, axis=0)
        exp_vals = tl.exp(logits_scaled - max_logit)
        sum_exp = tl.sum(exp_vals, axis=0)
        lse_val = max_logit + tl.log(sum_exp)  # natural log
        lse_base2 = lse_val / tl.log(2.0)  # base-2 logsumexp

        # Softmax over logits_scaled
        probs = exp_vals / sum_exp

        # Output vector for this head: out_vec = probs @ v_vals -> [HEAD_DIM]
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        for j in range(max_kv_idx):
            v_row = v_vals[j, :]  # [HEAD_DIM]
            out_vec += probs[j] * v_row

        # Store output[global_q_idx, h, :]
        out_base = out_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_base + tl.arange(0, HEAD_DIM), out_vec, mask=(tl.arange(0, HEAD_DIM) < HEAD_DIM))

        # Store lse[global_q_idx, h]
        lse_base = lse_ptr + global_q_idx * NUM_QO_HEADS + h
        tl.store(lse_base, lse_base2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda, "Triton requires CUDA tensors. Please move inputs to CUDA."
        assert q.dtype == torch.bfloat16, "q must be bfloat16"

        # Convert to float32 for compute; flatten k/v (squeeze the 1 dimension)
        q = q.contiguous().to(torch.float32)  # compute in fp32
        k_flat = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, NUM_KV_HEADS, HEAD_DIM]
        v_flat = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, NUM_KV_HEADS, HEAD_DIM]

        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        total_q = q.shape[0]
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs in fp32 for stability
        output_fp32 = torch.empty((total_q, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse_fp32 = torch.empty((total_q, NUM_QO_HEADS), dtype=torch.float32, device=q.device)

        # Iterate over batch elements b from 0 to len_indptr-2
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            num_q_tokens_in_b = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_indices_in_b = kv_end - kv_start

            # Launch Triton kernel once per (b, q_idx), looping over all 32 heads inside the kernel
            grid = (1, num_q_tokens_in_b)
            process_b_q[grid](
                q, k_flat, v_flat, output_fp32, lse_fp32, qo_indptr, kv_indptr, kv_indices,
                len_indptr, total_q, num_q_tokens_in_b, num_kv_indices_in_b, sm_scale
            )

        # Cast output to bfloat16 to match original
        output = output_fp32.to(torch.bfloat16)
        lse = lse_fp32  # fp32
        return output, lse


def run(*args):
    return ModelNew()(*args)
