import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_single_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    v_stride_b, v_stride_h, v_stride_d,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
    scale: tl.float32,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Initialize running max and sum for logsumexp
    neg_inf = -float("inf")
    m = neg_inf
    s = 0.0

    # Loop over kv positions j
    for j in range(0, num_kv_tokens):
        # Load q[i, h]
        q_offset = i * q_stride_b + h * q_stride_h
        q_val = tl.load(q_ptr + q_offset)

        # Load k_expanded[j, h]
        k_offset = j * k_stride_b + h * k_stride_h
        k_val = tl.load(k_ptr + k_offset)

        # Compute score
        score = q_val * k_val * scale

        # Apply causal mask: i can only attend to j < i + 1 + delta
        delta = num_kv_tokens - num_q_tokens
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, neg_inf)

        # Update logsumexp components
        m_new = tl.maximum(m, score)
        s = s * tl.exp(m - m_new) + tl.exp(score - m_new)
        m = m_new

    # Compute LSE per (i, h): logsumexp divided by log(2.0)
    inv_log2 = 1.0 / math.log(2.0)
    lse_val = tl.log(s) * inv_log2

    # Store LSE
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)

    # Accumulate output for each j again with final m, using same loop
    # We recompute scores and accumulate output[i, h, :] += y * v_expanded[j, h, :]
    out_base = i * out_stride_b + h * out_stride_h
    d = tl.arange(0, head_dim)  # feature dimension vector
    out_ptr_vec = out_ptr + out_base + d

    # Initialize output vector to zeros
    zeros_vec = tl.zeros((head_dim,), dtype=tl.float32)
    tl.store(out_ptr_vec, zeros_vec)

    for j in range(0, num_kv_tokens):
        # Load q[i, h]
        q_offset = i * q_stride_b + h * q_stride_h
        q_val = tl.load(q_ptr + q_offset)

        # Load k_expanded[j, h]
        k_offset = j * k_stride_b + h * k_stride_h
        k_val = tl.load(k_ptr + k_offset)

        # Compute score
        score = q_val * k_val * scale

        # Causal mask
        delta = num_kv_tokens - num_q_tokens
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, neg_inf)

        # Compute softmax factor y = exp(score - m)
        y = tl.exp(score - m)

        # v_expanded[j, h, :] vector along head_dim
        v_vec_base = j * v_stride_b + h * v_stride_h
        v_ptr_vec = v_ptr + v_vec_base + d
        v_vec = tl.load(v_ptr_vec)

        # out[i, h, :] += y * v_vec
        out_vals = tl.load(out_ptr_vec)
        out_vals = out_vals + y * v_vec
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

        # Extract sequence lengths
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        device = q.device

        # Output and LSE for full sequence
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

            # Allocate output accumulator and LSE for this slice
            out_acc = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            lse_slice = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)

            # Strides for Triton
            q_stride_b = q_slice.stride(0) * head_dim * q_slice.shape[1]  # stride in elements for batch dim
            q_stride_h = q_slice.stride(1) * head_dim                    # stride for head dim
            q_stride_d = q_slice.stride(2)                               # feature stride (dim=2)

            k_stride_b = k_expanded.stride(0) * head_dim * k_expanded.shape[1]
            k_stride_h = k_expanded.stride(1) * head_dim
            k_stride_d = k_expanded.stride(2)

            v_stride_b = v_expanded.stride(0) * head_dim * v_expanded.shape[1]
            v_stride_h = v_expanded.stride(1) * head_dim
            v_stride_d = v_expanded.stride(2)

            out_stride_b = out_acc.stride(0) * head_dim * out_acc.shape[1]
            out_stride_h = out_acc.stride(1) * head_dim
            out_stride_d = out_acc.stride(2)

            lse_stride_b = lse_slice.stride(0) * lse_slice.shape[1]
            lse_stride_h = lse_slice.stride(1)

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _attention_single_kernel[grid](
                q_slice, k_expanded, v_expanded, out_acc, lse_slice,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                v_stride_b, v_stride_h, v_stride_d,
                out_stride_b, out_stride_h, out_stride_d,
                lse_stride_b, lse_stride_h,
                sm_scale,
            )

            # Store results into output for this b-slice
            output[q_start:q_end] = out_acc.to(torch.bfloat16)
            lse[q_start:q_end] = lse_slice

        return output, lse


def run(*args):
    return ModelNew()(*args)
