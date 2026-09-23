import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_single_kernel(
    q_ptr, k_ptr, v_exp_ptr, out_ptr, lse_ptr,
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

    # Compute LSE per (i, h) using max-shift trick
    delta = num_kv_tokens - num_q_tokens  # runtime scalar
    m = -float("inf")
    s = 0.0

    for j in range(0, num_kv_tokens):
        # Load q[i, h] and k_expanded[j, h]
        q_offset = i * q_stride_b + h * q_stride_h
        k_offset = j * k_stride_b + h * k_stride_h
        q_val = tl.load(q_ptr + q_offset)
        k_val = tl.load(k_ptr + k_offset)

        # Compute score
        score = q_val * k_val * scale

        # Causal mask: only j < (i + 1 + delta)
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, -float("inf"))

        m = tl.maximum(m, score)
        s += tl.exp(score - m)

    # LSE per (i, h): logsumexp across j, divided by log(2.0) to match original behavior
    inv_log2 = 1.0 / math.log(2.0)
    lse_val = tl.log(s) * inv_log2

    # Store lse[i, h]
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)

    # Accumulate output for each j with causal mask
    out_base = i * out_stride_b + h * out_stride_h
    d = tl.arange(0, head_dim)  # vector along feature dim (128)
    out_ptr_vec = out_ptr + out_base + d

    # Initialize output vector to zeros
    zeros_vec = tl.zeros((head_dim,), dtype=tl.float32)
    tl.store(out_ptr_vec, zeros_vec)

    for j in range(0, num_kv_tokens):
        # Load q[i, h] and k_expanded[j, h]
        q_offset = i * q_stride_b + h * q_stride_h
        k_offset = j * k_stride_b + h * k_stride_h
        q_val = tl.load(q_ptr + q_offset)
        k_val = tl.load(k_ptr + k_offset)

        # Compute score
        score = q_val * k_val * scale

        # Causal mask
        causal = j < (i + 1 + delta)
        if not causal:
            continue  # no contribution

        y = tl.exp(score - lse_val)  # softmax factor

        # v_expanded[j, h, :] vector along head_dim
        v_base = j * v_stride_b + h * v_stride_h
        v_ptr_vec = v_exp_ptr + v_base + d

        # Load v and accumulate
        v_vals = tl.load(v_ptr_vec)
        out_vals = tl.load(out_ptr_vec) + y * v_vals
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

        # Output and LSE for the whole sequence
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

            # Allocate output buffer [num_q_tokens, 32, 128], float32
            out_acc = torch.zeros((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)

            # Strides for Triton
            q_stride_b = q_slice.stride(0) * head_dim * q_slice.shape[1]  # elements between batches
            q_stride_h = q_slice.stride(1) * head_dim                    # elements between heads
            q_stride_d = q_slice.stride(2)                               # elements between features

            k_stride_b = k_expanded.stride(0) * head_dim * k_expanded.shape[1]
            k_stride_h = k_expanded.stride(1) * head_dim
            k_stride_d = k_expanded.stride(2)

            v_stride_b = v_expanded.stride(0) * head_dim * v_expanded.shape[1]
            v_stride_h = v_expanded.stride(1) * head_dim
            v_stride_d = v_expanded.stride(2)

            out_stride_b = out_acc.stride(0) * head_dim * out_acc.shape[1]
            out_stride_h = out_acc.stride(1) * head_dim
            out_stride_d = out_acc.stride(2)

            lse_stride_b = lse.stride(0) * lse.shape[1]
            lse_stride_h = lse.stride(1)

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _attention_single_kernel[grid](
                q_slice, k_expanded, v_expanded, out_acc, lse,
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
            # Store lse for this b-slice
            lse[q_start:q_end] = lse[:num_q_tokens]  # lse tensor for this slice is the first num_q_tokens rows

        return output, lse


def run(*args):
    return ModelNew()(*args)
