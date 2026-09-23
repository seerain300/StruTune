import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_per_j_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    logits_stride_q, logits_stride_h, logits_stride_j,
    scale: tl.float32,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Tile over kv positions with fixed BLOCK_J; here head_dim == 128, so single tile
    BLOCK_J = 128
    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_idx = j0 + tl.arange(0, BLOCK_J)  # vector of j indices
        mask_j = j_idx < num_kv_tokens

        # Load q[i, h] scalar
        q_offset = i * q_stride_b + h * q_stride_h
        q_val = tl.load(q_ptr + q_offset)

        # Load k_expanded[j, h] as vector (masked)
        k_offset = j_idx * k_stride_b + h * k_stride_h
        k_val = tl.load(k_ptr + k_offset, mask=mask_j, other=0.0)

        # Compute score vector: q_val * k_val * scale
        score = q_val * k_val * scale

        # Causal mask: j < (i + 1 + delta), where delta = num_kv_tokens - num_q_tokens
        delta = num_kv_tokens - num_q_tokens
        causal_mask = j_idx < (i + 1 + delta)
        score = tl.where(causal_mask & mask_j, score, -float("inf"))

        # Store logits[i, h, j] as float32 (masked)
        logits_offset = i * logits_stride_q + h * logits_stride_h + j_idx * logits_stride_j
        tl.store(logits_ptr + logits_offset, score, mask=mask_j)


@triton.jit
def _lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_b, lse_stride_h,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Compute logsumexp along j for (i, h): lse[i, h] = log(sum(exp(score - m))) + m
    # Pass 1: m = max(score)
    m = -float("inf")
    BLOCK_J = 128
    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_idx = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_idx < num_kv_tokens
        offset = i * lse_stride_b + h * lse_stride_h + j_idx
        score = tl.load(logits_ptr + offset, mask=mask_j, other=-float("inf"))
        m = tl.maximum(m, tl.max(score, axis=0))

    # Pass 2: s = sum(exp(score - m))
    s = 0.0
    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_idx = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_idx < num_kv_tokens
        offset = i * lse_stride_b + h * lse_stride_h + j_idx
        score = tl.load(logits_ptr + offset, mask=mask_j, other=-float("inf"))
        y = tl.exp(score - m)
        # sum along the vector
        s += tl.sum(y, axis=0)

    # Store lse[i, h] = log(s) + m (standard logsumexp normalization)
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, tl.log(s) + m)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, lse_ptr, v_expanded_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_offset = i * lse_stride_b + h * lse_stride_h
    m = tl.load(lse_ptr + lse_offset)

    # Accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - m) * v_expanded[j, h, :]
    out_base = i * out_stride_b + h * out_stride_h
    d = tl.arange(0, 128)  # along head_dim
    out_ptr_vec = out_ptr + out_base + d  # vector along feature dim

    # Initialize out vector
    out_vals = tl.load(out_ptr_vec)

    BLOCK_J = 128
    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_idx = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_idx < num_kv_tokens

        # score = logits[i, h, j]
        offset = i * lse_stride_b + h * lse_stride_h + j_idx
        score = tl.load(logits_ptr + offset, mask=mask_j, other=-float("inf"))
        y = tl.exp(score - m)  # vector

        # v_expanded[j, h, :] is [128] contiguous, v_expanded shape [num_kv_tokens, num_qo_heads, 128]
        v_base = j_idx * (num_qo_heads * 128) + h * 128
        v_ptr_vec = v_expanded_ptr + v_base + d

        val = y[:, None] * tl.load(v_ptr_vec)  # [BLOCK_J, 128]
        # Reduce over j: sum over first dim -> [128]
        contribution = tl.sum(val, axis=0)

        # Accumulate
        out_vals = out_vals + contribution
        tl.store(out_ptr_vec, out_vals)

    # Final out stored; out_ptr_vec is updated each iteration


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton requires CUDA tensors"

        # Constants from original code
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        device = q.device

        # Output and LSE tensors (full size), will be written per b-slice
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment (b-slice)
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice tensors
            q_slice = q[q_start:q_end].to(torch.float32).contiguous()  # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].to(torch.float32).contiguous()  # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].to(torch.float32).contiguous()  # [num_kv_tokens, 8, 128]

            # Expand K/V to 32 heads (GQA)
            k_expanded = k_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_expanded.shape[0]

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens], float32
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Strides for Triton
            q_stride_b = q_slice.stride(0) * head_dim * q_slice.shape[1]  # element stride for batch
            q_stride_h = q_slice.stride(1) * head_dim                    # element stride for head dim
            q_stride_d = q_slice.stride(2)                               # feature stride

            k_stride_b = k_expanded.stride(0) * head_dim * k_expanded.shape[1]
            k_stride_h = k_expanded.stride(1) * head_dim
            k_stride_d = k_expanded.stride(2)

            logits_stride_q = logits.stride(0) * num_kv_tokens
            logits_stride_h = logits.stride(1) * num_kv_tokens
            logits_stride_j = logits.stride(2)

            # Launch Triton kernel: grid = (num_q_tokens * num_qo_heads,), one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _compute_logits_per_j_kernel[grid](
                q_slice, k_expanded, logits,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                logits_stride_q, logits_stride_h, logits_stride_j,
                sm_scale,
            )

            # Compute lse per (i, h) using Triton reduction
            lse_slice = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)
            _lse_reduce_kernel[grid](
                logits, lse_slice,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                lse_slice.stride(0), lse_slice.stride(1),
            )

            # Accumulate output per (i, h) using Triton softmax*attention accumulation
            out_acc = torch.zeros((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            out_stride_b = out_acc.stride(0) * head_dim * out_acc.shape[1]
            out_stride_h = out_acc.stride(1) * head_dim
            out_stride_d = out_acc.stride(2)

            _softmax_accum_output_kernel[grid](
                logits, lse_slice, v_expanded, out_acc,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                out_stride_b, out_stride_h, out_stride_d,
                lse_slice.stride(0), lse_slice.stride(1),
            )

            # Store results into output for this b-slice (cast to bfloat16)
            output[q_start:q_end] = out_acc.to(torch.bfloat16)
            # Store lse for this b-slice (float32)
            lse[q_start:q_end] = lse_slice

        # Return final output and lse (already filled per slice)
        return output, lse


def run(*args):
    return ModelNew()(*args)
