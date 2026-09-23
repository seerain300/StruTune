import torch
import math

# Triton kernels

@triton.jit
def _compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens,
    sm_scale,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d,
    stride_log_b, stride_log_h, stride_log_d,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32  # num_qo_heads = 32 (fixed)
    h = pid % 32

    if i >= num_q_tokens:
        return

    # Compute q offset and load q[i, h]
    q_off = i * stride_q_b + h * stride_q_h
    q_val = tl.load(q_ptr + q_off)

    # Loop over j (KV positions)
    for j in range(0, num_kv_tokens):
        k_off = j * stride_k_b + h * stride_k_h
        k_val = tl.load(k_ptr + k_off)

        score = q_val * k_val * sm_scale  # scalar float32

        # Store logits[i, h, j]
        log_off = i * stride_log_b + h * stride_log_h + j * stride_log_d
        tl.store(logits_ptr + log_off, score)


@triton.jit
def _lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens,
    stride_log_b, stride_log_h, stride_log_d,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    m = -float('inf')
    s = 0.0

    for j in range(0, num_kv_tokens):
        log_off = i * stride_log_b + h * stride_log_h + j * stride_log_d
        val = tl.load(logits_ptr + log_off)  # scalar
        m_new = tl.maximum(m, val)
        s = s * tl.exp(m - m_new) + tl.exp(val - m_new)
        m = m_new

    lse_val = tl.log(s) + m  # standard logsumexp
    lse_off = i * 32 + h     # since lse has shape [num_q_tokens, 32]
    tl.store(lse_ptr + lse_off, lse_val)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, v_ptr, lse_ptr, output_ptr,
    num_q_tokens, num_kv_tokens, delta,
    stride_log_b, stride_log_h, stride_log_d,
    stride_v_b, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    head_dim: tl.constexpr,  # 128
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    out_base = i * stride_out_b + h * stride_out_h
    d = tl.arange(0, head_dim)  # vector of 128
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    # Accumulate over j with causal mask j < (i + 1 + delta)
    for j in range(0, num_kv_tokens):
        valid = j < (i + 1 + delta)
        log_off = i * stride_log_b + h * stride_log_h + j * stride_log_d
        val = tl.load(logits_ptr + log_off)  # scalar logits[i, h, j]
        prob = tl.exp(val - tl.load(lse_ptr + (i * 32 + h)))  # softmax prob
        # If invalid, zero out
        prob = tl.where(valid, prob, 0.0)

        # Load v_expanded[j, h, :]
        v_off = j * stride_v_b + h * stride_v_h
        v_vec = tl.load(v_ptr + v_off + d)  # vector [128]

        out_vec += prob * v_vec

    # Store output[i, h, :]
    tl.store(output_ptr + out_base + d, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda
        total_q = q.shape[0]
        total_kv = k.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Prepare output and lse
        output = torch.empty((total_q, num_qo_heads, 128), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        len_indptr = qo_indptr.numel()
        b = 0
        while b < len_indptr - 1:
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                b += 1
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice and convert to float32
            q_slice = q[q_start:q_end].contiguous().to(torch.float32)     # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].contiguous().to(torch.float32)   # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].contiguous().to(torch.float32)   # [num_kv_tokens, 8, 128]

            # Expand k and v to 32 heads
            k_expanded = k_slice.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate logits buffer: [num_q_tokens, 32, num_kv_tokens]
            logits = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Launch kernel to compute logits
            grid = (num_q_tokens * 32,)
            _compute_logits_kernel[grid](
                q_slice, k_expanded, logits,
                num_q_tokens, num_kv_tokens,
                sm_scale,
                q_slice.stride(0), q_slice.stride(1), q_slice.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2),
            )

            # Launch kernel to compute LSE per (i, h)
            grid_lse = (num_q_tokens * 32,)
            lse_partial = torch.empty((num_q_tokens * 32,), dtype=torch.float32, device=q.device)
            _lse_reduce_kernel[grid_lse](
                logits, lse_partial,
                num_q_tokens, num_kv_tokens,
                logits.stride(0), logits.stride(1), logits.stride(2),
            )
            lse[b * num_q_tokens : (b + 1) * num_q_tokens, :] = lse_partial.view(num_q_tokens, 32)

            # Launch kernel to compute output[i, h, :]
            output_seg = output[q_start:q_end]  # [num_q_tokens, 32, 128], bfloat16
            delta = num_kv_tokens - num_q_tokens

            _softmax_accum_output_kernel[grid](
                logits, v_expanded, lse[b * num_q_tokens : (b + 1) * num_q_tokens], output_seg,
                num_q_tokens, num_kv_tokens, delta,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                head_dim=128,
            )

            b += 1

        return output, lse


def run(*args):
    return ModelNew()(*args)
