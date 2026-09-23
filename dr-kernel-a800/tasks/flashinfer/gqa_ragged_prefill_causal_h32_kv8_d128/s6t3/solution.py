import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_per_j_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
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

    # Loop over kv positions and compute q[i, h] * k_expanded[j, h] * scale, store in logits[i, h, j]
    for j in range(0, num_kv_tokens):
        # Load q[i, h] scalar
        q_offset = i * q_stride_b + h * q_stride_h
        q_val = tl.load(q_ptr + q_offset)

        # Load k_expanded[j, h] scalar
        k_offset = j * k_stride_b + h * k_stride_h
        k_val = tl.load(k_ptr + k_offset)

        # Compute score
        score = q_val * k_val * scale

        # Causal mask: i can only attend to j < i + 1 + delta
        delta = num_kv_tokens - num_q_tokens  # runtime scalar
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, -float("inf"))

        # Store logits[i, h, j] in float32
        logits_offset = i * logits_stride_q + h * logits_stride_h + j * logits_stride_j
        tl.store(logits_ptr + logits_offset, score)


@triton.jit
def _lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_b, lse_stride_h,
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Compute logsumexp along j for (i, h): lse[i, h] = log(sum(exp(score - m))) + m
    # Pass 1: m = max(score)
    m = -float("inf")
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        m = tl.maximum(m, score)

    # Pass 2: s = sum(exp(score - m))
    s = 0.0
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)
        s += y

    # Store lse[i, h] = log(s) + m
    lse_val = tl.log(s) + m
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, lse_ptr, v_expanded_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
):
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
    d = tl.arange(0, 128)
    out_ptr_vec = out_ptr + out_base + d

    for j in range(0, num_kv_tokens):
        # score = q[i, h] * k_expanded[j, h] * scale = logits[i, h, j]
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)

        # v_expanded[j, h, :]
        # v_expanded has shape [num_kv_tokens, num_qo_heads, head_dim] = [N, 32, 128], contiguous
        # base offset for v_expanded[j, h, :] is j * (32 * 128) + h * 128
        v_base = j * (num_qo_heads * head_dim) + h * head_dim
        v_ptr_vec = v_expanded_ptr + v_base + d
        val = y * tl.load(v_ptr_vec)

        # out[i, h, :] += val
        out_vals = tl.load(out_ptr_vec)
        out_vals = out_vals + val
        tl.store(out_ptr_vec, out_vals)


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
        gqa_ratio = num_qo_heads // num_kv_heads

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        device = q.device

        # Output and LSE
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


def run(*args):
    return ModelNew()(*args)
