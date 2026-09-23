import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_bf16_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens,
    num_qo_heads, head_dim,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    v_stride_b, v_stride_h, v_stride_d,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
    sm_scale,
    delta,  # num_kv_tokens - num_q_tokens (int32)
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads

    if i >= num_q_tokens or h >= num_qo_heads:
        return

    # Initialize running max and sum for logsumexp (stable accumulation)
    m = -float('inf')
    s = 0.0

    # First pass: compute LSE and accumulate output
    for j in range(0, num_kv_tokens):
        # Load q[i, h], k[j, h], v[j, h] as scalars
        q_off = i * q_stride_b + h * q_stride_h + 0 * q_stride_d
        k_off = j * k_stride_b + h * k_stride_h + 0 * k_stride_d
        v_off = j * v_stride_b + v_stride_h * h + 0 * v_stride_d

        q_ih = tl.load(q_ptr + q_off).to(tl.float32)
        k_jh = tl.load(k_ptr + k_off).to(tl.float32)
        v_jh = tl.load(v_ptr + v_off).to(tl.float32)

        # score = q[i, h] * k[j, h] * sm_scale
        score = q_ih * k_jh * sm_scale

        # Apply causal mask: j < (i + 1 + delta)
        mask = (j < (i + 1 + delta))
        score = tl.where(mask, score, -float('inf'))

        # Update running max and sum
        m_new = tl.maximum(m, score)
        s = s * tl.exp(m - m_new) + tl.exp(score - m_new)
        m = m_new

    # Compute LSE per (i, h): log(s) / ln(2)
    inv_log2 = 1.0 / 0.6931471805599453  # 1.0 / ln(2)
    lse_val = tl.log(s) * inv_log2

    # Store LSE[i, h]
    lse_off = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: recompute attn and accumulate output
    for j in range(0, num_kv_tokens):
        q_off = i * q_stride_b + h * q_stride_h + 0 * q_stride_d
        k_off = j * k_stride_b + h * k_stride_h + 0 * k_stride_d
        v_off = j * v_stride_b + v_stride_h * h + 0 * v_stride_d

        q_ih = tl.load(q_ptr + q_off).to(tl.float32)
        k_jh = tl.load(k_ptr + k_off).to(tl.float32)
        v_jh = tl.load(v_ptr + v_off).to(tl.float32)

        score = q_ih * k_jh * sm_scale
        mask = (j < (i + 1 + delta))
        score = tl.where(mask, score, -float('inf'))

        attn = tl.exp(score - lse_val)

        # Accumulate output[i, h, :] += attn * v[j, h, :]
        out_off = i * out_stride_b + h * out_stride_h + 0 * out_stride_d
        # Vectorize across head_dim (d=0..127)
        for d in range(0, head_dim):
            val = attn * v_jh  # v_jh is scalar; broadcast over d
            tl.store(out_ptr + out_off + d * out_stride_d, val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        device = q.device

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]

        # Prepare global output and LSE
        output = torch.empty((total_q, 32, 128), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Iterate over segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # Skip empty segments
                continue

            # Slice
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            # Cast to float32 for computation
            q_batch = q_batch.to(torch.float32)
            k_batch = k_batch.to(torch.float32)
            v_batch = v_batch.to(torch.float32)

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]
            gqa_ratio = 32 // 8  # 4
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate per-segment output and lse
            out_acc = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=device)

            # Strides
            q_strides = q_batch.stride()         # [32, 128, 1] for contiguous
            k_strides = k_expanded.stride()      # [32, 128, 1] for contiguous
            v_strides = v_expanded.stride()      # [32, 128, 1] for contiguous
            out_acc_strides = out_acc.stride()   # [32, 128, 1] for contiguous

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * 32,)

            _attention_bf16_kernel[grid](
                q_batch, k_expanded, v_expanded, out_acc, lse_seg,
                num_q_tokens, num_kv_tokens,
                32, 128,
                q_strides[0], q_strides[1], q_strides[2],
                k_strides[0], k_strides[1], k_strides[2],
                v_strides[0], v_strides[1], v_strides[2],
                out_acc_strides[0], out_acc_strides[1], out_acc_strides[2],
                lse_seg.stride(0), lse_seg.stride(1),
                sm_scale,
                num_kv_tokens - num_q_tokens,  # delta
                num_warps=1, num_stages=1
            )

            # Store into global output and lse for this segment
            output[q_start:q_start + num_q_tokens, :, :] = out_acc.to(torch.bfloat16)
            lse[q_start:q_start + num_q_tokens, :] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
