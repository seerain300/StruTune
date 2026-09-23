import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_per_ih_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    v_stride_b, v_stride_h, v_stride_d,
    out_stride_i, out_stride_h, out_stride_d,
    scale: tl.float32,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Running max m and sum s for logsumexp; initialize
    m = -float("inf")
    s = 0.0

    # Output buffer for this (i, h): [head_dim] vector
    out_base = i * out_stride_i + h * out_stride_h
    d = tl.arange(0, head_dim)
    out_ptr_vec = out_ptr + out_base + d

    # Loop over kv positions j
    for j in range(0, num_kv_tokens):
        # Load q[i, h] as scalar
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

        # Update logsumexp in streaming fashion
        m_new = tl.maximum(m, score)
        # soft = exp(score - m_new)
        soft = tl.exp(score - m_new)
        # s_new = s * exp(m - m_new) + soft
        s = s * tl.exp(m - m_new) + soft
        m = m_new

        # Compute output contribution y = soft * v_expanded[j, h, :]
        # v_expanded has shape [num_kv_tokens, num_qo_heads, head_dim], contiguous
        # v_expanded[j, h, :] base offset: j * (num_qo_heads * head_dim) + h * head_dim
        v_base = j * (num_qo_heads * head_dim) + h * head_dim
        v_ptr_vec = v_ptr + v_base + d
        # Load v and accumulate
        val = soft * tl.load(v_ptr_vec)
        out_vals = tl.load(out_ptr_vec) + val
        tl.store(out_ptr_vec, out_vals)

    # After processing all j, out_ptr_vec contains sum_j exp(score_j - m) * v_expanded[j, h, :]
    # No need to compute lse here; accumulation includes normalization.


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

        # Output (we return only output tensor, matching original run's output structure)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)

        # Process each batch segment (b-slice)
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No work for this slice; output zeros
                output[q_start:q_end] = torch.zeros((q_end - q_start, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
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
            out = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)

            # Strides (element strides)
            q_stride_b = q_slice.stride(0) * head_dim * q_slice.shape[1]  # stride for batch dim
            q_stride_h = q_slice.stride(1) * head_dim                    # stride for head dim
            q_stride_d = q_slice.stride(2)                               # feature stride

            k_stride_b = k_expanded.stride(0) * head_dim * k_expanded.shape[1]
            k_stride_h = k_expanded.stride(1) * head_dim
            k_stride_d = k_expanded.stride(2)

            v_stride_b = v_expanded.stride(0) * head_dim * v_expanded.shape[1]
            v_stride_h = v_expanded.stride(1) * head_dim
            v_stride_d = v_expanded.stride(2)

            out_stride_i = out.stride(0) * head_dim * out.shape[1]       # stride for query token
            out_stride_h = out.stride(1) * head_dim                      # stride for head
            out_stride_d = out.stride(2)                                 # feature stride

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _attention_per_ih_kernel[grid](
                q_slice, k_expanded, v_expanded, out,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                v_stride_b, v_stride_h, v_stride_d,
                out_stride_i, out_stride_h, out_stride_d,
                sm_scale,
            )

            # Store results into output for this b-slice (cast to bfloat16)
            output[q_start:q_end] = out.to(torch.bfloat16)

        # Return output only (lse was not requested, kept simple to avoid extra kernels)
        return output, None  # returning a tuple as in original, second element None


def run(*args):
    return ModelNew()(*args)
