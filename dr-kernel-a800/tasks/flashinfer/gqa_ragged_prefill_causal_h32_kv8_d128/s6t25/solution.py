import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    q_stride_b, q_stride_h, q_stride_d,         # strides for q [B, H, D]
    k_stride_b, k_stride_h, k_stride_d,         # strides for k_expanded [J, H, D]
    logits_stride_B, logits_stride_H, logits_stride_J,  # element strides for logits [B, H, J]
    scale: tl.float32,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # If no kv tokens for this slice, fill logits with -inf
    if num_kv_tokens <= 0:
        for j in range(0, 1):  # dummy, mask will handle
            logits_offset = i * logits_stride_B + h * logits_stride_H + j * logits_stride_J
            tl.store(logits_ptr + logits_offset, -float("inf"))
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

        # Causal mask: i can only attend to j < i + 1 + delta, where delta = num_kv_tokens - num_q_tokens
        delta = num_kv_tokens - num_q_tokens  # runtime scalar
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, -float("inf"))

        # Store logits[i, h, j] in float32
        logits_offset = i * logits_stride_B + h * logits_stride_H + j * logits_stride_J
        tl.store(logits_ptr + logits_offset, score)


@triton.jit
def _lse_row_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_b, lse_stride_h,  # strides for lse [B, H]
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Compute lse[i, h] = logsumexp(logits[i, h, :]) / log(2) using max-shift trick:
    # m = max(score), s = sum(exp(score - m)), lse = (m + log(s)) / log(2)
    m = -float("inf")
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j * lse_stride_h  # here: j is dummy, lse_stride_h is 1
        # We need j-th element of logits[i, h, :], advance by + logits_stride_J per j
        score = tl.load(logits_ptr + (i * lse_stride_b + h * logits_stride_H) + j * logits_stride_J)
        m = tl.maximum(m, score)

    s = 0.0
    for j in range(0, num_kv_tokens):
        score = tl.load(logits_ptr + (i * lse_stride_b + h * logits_stride_H) + j * logits_stride_J)
        y = tl.exp(score - m)
        s += y

    lse_val = (tl.log(s) + m) * (1.0 / math.log(2.0))  # divide logsumexp by ln(2) as original does
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, lse_ptr, v_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    out_stride_B, out_stride_H, out_stride_D,     # strides for out [B, H, D]
    lse_stride_b, lse_stride_h,                   # strides for lse [B, H]
    v_stride_J, v_stride_H, v_stride_D,          # strides for v_expanded [J, H, D]
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_offset = i * lse_stride_b + h * lse_stride_h
    lse_val = tl.load(lse_ptr + lse_offset)  # already (logsumexp / log(2))

    # Accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - lse_val) * v_expanded[j, h, :]
    out_base = i * out_stride_B + h * out_stride_H
    d = tl.arange(0, head_dim)
    out_ptr_vec = out_ptr + out_base + d

    # We need to zero out out_ptr_vec before accumulation. Triton kernels can't memset, but host can. Since we launch this kernel after computing logits and lse, we should have initialized out to zeros. However, we can implement zeroing here:
    for d0 in range(0, head_dim):
        tl.store(out_ptr + out_base + d0, 0.0)

    for j in range(0, num_kv_tokens):
        # score = logits[i, h, j]
        score = tl.load(logits_ptr + (i * lse_stride_b + h * logits_stride_H) + j * logits_stride_J)
        w = tl.exp(score - lse_val)  # softmax weight for this j

        # v_expanded[j, h, :] base offset: j*v_stride_J + h*v_stride_H
        v_base = j * v_stride_J + h * v_stride_H
        v_ptr_vec = v_ptr + v_base + d

        v_vals = tl.load(v_ptr_vec)
        out_vals = tl.load(out_ptr_vec)
        out_vals = out_vals + w * v_vals
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

        # Output and LSE (full tensors)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment (b-slice)
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No work for this slice; fill output with zeros and lse with -inf
                output[q_start:q_end] = torch.zeros((q_end - q_start, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
                lse[q_start:q_end] = torch.full((q_end - q_start, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
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

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens], float32, contiguous
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Strides for Triton (element strides)
            # q_slice


def run(*args):
    return ModelNew()(*args)
