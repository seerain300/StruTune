import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_b_slice_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    v_stride_b, v_stride_h, v_stride_d,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
    scale: tl.float32,
    BLOCK_D: tl.constexpr,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Running max and sum for logsumexp per (i, h)
    neg_inf = -float("inf")
    m = tl.full((), neg_inf, tl.float32)
    s = tl.full((), 0.0, tl.float32)

    # Iterate over kv positions in chunks
    for j0 in range(0, num_kv_tokens, BLOCK_D):
        # Compute per-chunk logsumexp (with causal mask)
        for jj in range(0, BLOCK_D):
            j = j0 + jj
            mask_j = j < num_kv_tokens

            # Load q[i, h] (scalar)
            q_offset = i * q_stride_b + h * q_stride_h
            q_val = tl.load(q_ptr + q_offset, mask=mask_j, other=0.0)

            # Load k_expanded[j, h] and v_expanded[j, h] (scalars)
            k_offset = j * k_stride_b + h * k_stride_h
            v_offset = j * v_stride_b + h * v_stride_h
            k_val = tl.load(k_ptr + k_offset, mask=mask_j, other=0.0)
            v_val = tl.load(v_ptr + v_offset, mask=mask_j, other=0.0)

            # Score
            score = q_val * k_val * scale

            # Causal mask: i can only attend to j < i + 1 + delta
            delta = num_kv_tokens - num_q_tokens
            causal = (j < (i + 1 + delta)) and mask_j
            score = tl.where(causal, score, neg_inf)

            # Accumulate logsumexp
            y = tl.exp(score - m)
            m_new = tl.maximum(m, score)
            s = s * tl.exp(m - m_new) + y
            m = m_new

        # After processing the chunk, compute output contributions for each j in the chunk
        for jj in range(0, BLOCK_D):
            j = j0 + jj
            if j >= num_kv_tokens:
                break

            # Recompute score with causal mask
            q_offset = i * q_stride_b + h * q_stride_h
            q_val = tl.load(q_ptr + q_offset)
            k_offset = j * k_stride_b + h * k_stride_h
            v_offset = j * v_stride_b + h * v_stride_h
            k_val = tl.load(k_ptr + k_offset)
            v_val = tl.load(v_ptr + v_offset)

            score = q_val * k_val * scale
            delta = num_kv_tokens - num_q_tokens
            causal = (j < (i + 1 + delta))
            if not causal:
                score = neg_inf

            # Softmax factor for this j
            y = tl.exp(score - m)

            # Accumulate output[i, h, :] += y * v_expanded[j, h, :]
            # out is [num_q_tokens, num_qo_heads, head_dim] float32
            out_base = i * out_stride_b + h * out_stride_h
            # v_expanded[j, h, :] is [head_dim]
            v_vec_base = j * v_stride_b + h * v_stride_h
            # Vectorized along head_dim
            d = tl.arange(0, head_dim)
            out_ptr_vec = out_ptr + out_base + d
            v_ptr_vec = v_ptr + v_vec_base + d

            # Load current output, multiply-add, store
            out_vals = tl.load(out_ptr_vec)
            out_vals = out_vals + y * tl.load(v_ptr_vec)
            tl.store(out_ptr_vec, out_vals)

    # Compute LSE and store: lse[i, h] = log(s) / log(2)
    inv_log2 = 1.0 / math.log(2.0)
    lse_val = tl.log(s) * inv_log2
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)


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

        # Process each batch segment
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

            # Expand K/V to match 32 heads (GQA)
            k_expanded = k_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_expanded.shape[0]

            # Allocate output for this b-slice (float32, cast to bfloat16 after kernel)
            out = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)

            # Strides (element strides)
            q_stride_b = q_slice.stride(0) * head_dim * q_slice.shape[1]
            q_stride_h = q_slice.stride(1) * head_dim
            q_stride_d = q_slice.stride(2)

            k_stride_b = k_expanded.stride(0) * head_dim * k_expanded.shape[1]
            k_stride_h = k_expanded.stride(1) * head_dim
            k_stride_d = k_expanded.stride(2)

            v_stride_b = v_expanded.stride(0) * head_dim * v_expanded.shape[1]
            v_stride_h = v_expanded.stride(1) * head_dim
            v_stride_d = v_expanded.stride(2)

            out_stride_b = out.stride(0) * head_dim * out.shape[1]
            out_stride_h = out.stride(1) * head_dim
            out_stride_d = out.stride(2)

            lse_stride_b = lse.stride(0) * lse.shape[1]
            lse_stride_h = lse.stride(1)

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _attention_b_slice_kernel[grid](
                q_slice, k_expanded, v_expanded, out, lse,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                v_stride_b, v_stride_h, v_stride_d,
                out_stride_b, out_stride_h, out_stride_d,
                lse_stride_b, lse_stride_h,
                sm_scale,  # scale = 1/sqrt(128)
                BLOCK_D=128,
            )

            # Store b-slice results into full output and lse
            output[q_start:q_end] = out.to(torch.bfloat16)
            # lse already matches [num_q_tokens, 32]; assign slice
            lse[q_start:q_end] = lse[:num_q_tokens]

        return output, lse


def run(*args):
    return ModelNew()(*args)
