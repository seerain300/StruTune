import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    logits_stride_i, logits_stride_h, logits_stride_j,
    scale: tl.float32,
):
    # Grid: (i, h, j) — one program per (i, h, j)
    i = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    j = tl.program_id(axis=2)
    if (i >= num_q_tokens) or (h >= num_qo_heads):
        return

    # Load q[i, h] scalar
    q_offset = i * q_stride_b + h * q_stride_h
    q_val = tl.load(q_ptr + q_offset)

    # Load k_expanded[j, h] scalar
    k_offset = j * k_stride_b + h * k_stride_h
    k_val = tl.load(k_ptr + k_offset)

    # Compute score
    score = q_val * k_val * scale

    # Causal mask: i can only attend to j < i + 1 + delta
    delta = num_kv_tokens - num_q_tokens
    causal = j < (i + 1 + delta)
    score = tl.where(causal, score, -float("inf"))

    # Store logits[i, h, j] in flattened buffer: index = i * (H * J) + h * J + j
    logits_index = i * logits_stride_i + h * logits_stride_h + j * logits_stride_j
    tl.store(logits_ptr + logits_index, score)


@triton.jit
def _lse_row_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_i, lse_stride_h,
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
        offset = i * lse_stride_i + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        m = tl.maximum(m, score)

    # Pass 2: s = sum(exp(score - m))
    s = 0.0
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_i + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)
        s += y

    # Multiply by inv_log2 to match original's division by log(2)
    inv_log2 = 1.0 / math.log(2.0)
    lse_val = tl.log(s) + m
    lse_offset = i * lse_stride_i + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val * inv_log2)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, lse_ptr, v_expanded_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    out_stride_i, out_stride_h, out_stride_d,
    lse_stride_i, lse_stride_h,
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load lse[i, h] (which is logsumexp * inv_log2)
    lse_offset = i * lse_stride_i + h * lse_stride_h
    m = tl.load(lse_ptr + lse_offset)

    # Accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - m) * v_expanded[j, h, :]
    out_base = i * out_stride_i + h * out_stride_h
    d = tl.arange(0, head_dim)
    out_ptr_vec = out_ptr + out_base + d

    for j in range(0, num_kv_tokens):
        # score = q[i, h] * k_expanded[j, h] * scale = logits[i, h, j]
        offset = i * lse_stride_i + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)

        # v_expanded[j, h, :]
        # v_expanded has shape [num_kv_tokens, num_qo_heads, head_dim], contiguous
        # Flattened index: j * (H * D) + h * D
        v_offset = j * (num_qo_heads * head_dim) + h * head_dim
        v_ptr_vec = v_expanded_ptr + v_offset + d
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

        # Constants
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

            # Handle empty slices: no work for this segment
            if q_start >= q_end or kv_start >= kv_end:
                # Write zeros for output and -inf for lse in this slice
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

            # Allocate flat buffers for logits, lse, and output
            # logits[i, h, j] flattened: index = i * (H * J) + h * J + j
            logits_flat = torch.empty((num_q_tokens * num_qo_heads * num_kv_tokens), dtype=torch.float32, device=device)
            lse_flat = torch.empty((num_q_tokens * num_qo_heads), dtype=torch.float32, device=device)
            out_flat = torch.empty((num_q_tokens * num_qo_heads * head_dim), dtype=torch.float32, device=device)

            # Compute element sizes (in bytes) and element strides
            # q: [N_q, 32, 128]
            q_stride_b = q_slice.stride(0) * q_slice.element_size()
            q_stride_h = q_slice.stride(1) * q_slice.element_size()
            q_stride_d = q_slice.stride(2) * q_slice.element_size()

            # k_expanded: [N_kv, 32, 128]
            k_stride_b = k_expanded.stride(0) * k_expanded.element_size()
            k_stride_h = k_expanded.stride(1) * k_expanded.element_size()
            k_stride_d = k_expanded.stride(2) * k_expanded.element_size()

            # logits: [N_q, 32, N_kv_tokens], flattened
            logits_stride_i = (num_qo_heads * num_kv_tokens) * q_slice.element_size()
            logits_stride_h = num_kv_tokens * q_slice.element_size()
            logits_stride_j = q_slice.element_size()

            # Launch Triton kernel to compute logits
            grid_logit = (num_q_tokens, num_qo_heads, num_kv_tokens)
            _compute_logits_kernel[grid_logit](
                q_slice, k_expanded, logits_flat,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                logits_stride_i, logits_stride_h, logits_stride_j,
                sm_scale,
            )

            # Launch Triton kernel to compute lse per (i, h)
            grid_lse = (num_q_tokens * num_qo_heads,)
            lse_stride_i = num_qo_heads * q_slice.element_size()
            lse_stride_h = q_slice.element_size()
            _lse_row_kernel[grid_lse](
                logits_flat, lse_flat,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                lse_stride_i, lse_stride_h,
            )

            # Launch Triton kernel to accumulate output[i, h, :]
            out_stride_i = (num_qo_heads * head_dim) * q_slice.element_size()
            out_stride_h = head_dim * q_slice.element_size()
            out_stride_d = q_slice.element_size()
            _softmax_accum_output_kernel[grid_lse](
                logits_flat, lse_flat, v_expanded, out_flat,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                out_stride_i, out_stride_h, out_stride_d,
                lse_stride_i, lse_stride_h,
            )

            # Store results into output for this b-slice (cast to bfloat16)
            out_acc = out_flat.view(num_q_tokens, num_qo_heads, head_dim)
            output[q_start:q_end] = out_acc.to(torch.bfloat16)

            # Store lse for this b-slice (float32), already matches shape
            lse[q_start:q_end] = lse_flat.view(num_q_tokens, num_qo_heads)

        return output, lse


def run(*args):
    return ModelNew()(*args)
