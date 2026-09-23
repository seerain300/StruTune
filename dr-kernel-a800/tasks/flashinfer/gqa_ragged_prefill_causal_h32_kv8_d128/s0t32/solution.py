import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_forward_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, q_end, kv_start, kv_end, sm_scale, delta,
):
    """
    Triton kernel: processes one segment.
    For each query position i and query head h, compute:
      - logits[j] = dot(q[i, h, :], k[kv_start + j, (h % 8), :]) * sm_scale for j in 0..7
      - apply forward-causal mask: logits[j] = -inf if j >= (i + 1 + delta)
      - lse[i, h] = logsumexp(logits) / ln(2)
      - softmax = exp(logits - lse)
      - out[i, h, :] += sum_j softmax[j] * v[kv_start + j, (h % 8), :]
    Stores output out_ptr[i * 32 + h, :] and lse lse_ptr[i * 32 + h].
    q_ptr: [q_end - q_start, 32, 128], float32
    k_ptr: [kv_end - kv_start, 8, 128], float32
    v_ptr: [kv_end - kv_start, 8, 128], float32
    out_ptr: [q_end - q_start, 32, 128], float32 (per (i,h) accumulator)
    lse_ptr: [q_end - q_start, 32], float32 (per (i,h))
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # Load q[i, h, :]
    q_lin = i * 32 * 128 + h * 128
    q_vec = tl.load(q_ptr + q_lin)  # [128] float32

    # Prepare logits and lse for this (i,h)
    logits = tl.zeros((8,), dtype=tl.float32)

    # Compute dot-products for j in 0..7 and apply causal mask
    for j in tl.static_range(8):
        orig_h = h % 8  # map to original kv head
        k_lin = (kv_start + j) * 8 * 128 + orig_h * 128
        k_vec = tl.load(k_ptr + k_lin)  # [128] float32
        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        # Forward-causal mask: if j >= (i + 1 + delta), set to -inf
        if j >= (i + 1 + delta):
            dot = -float("inf")
        logits[j] = dot

    # Base-2 logsumexp
    m = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - m), axis=0)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

    # Softmax across 8 positions
    soft = tl.exp(logits - lse_val)  # [8] float32

    # Accumulate output[i, h, :] += sum_j soft[j] * v[kv_start + j, orig_h, :]
    out_lin = i * 32 * 128 + h * 128
    # Initialize output vector
    out_vec = tl.load(out_ptr + out_lin)  # [128] float32
    for j in tl.static_range(8):
        orig_h = h % 8
        v_lin = (kv_start + j) * 8 * 128 + orig_h * 128
        v_vec = tl.load(v_ptr + v_lin)  # [128] float32
        out_vec += v_vec * soft[j]
    tl.store(out_ptr + out_lin, out_vec)

    # Store lse[i, h]
    lse_lin = i * 32
    tl.store(lse_ptr + lse_lin, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        sm_scale: float32
        Returns (output: [total_q, 32, 128] bfloat16, lse: [total_q, 32] float32 base-2)
        """
        # Ensure contiguous tensors and cast to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]

        # Prepare outputs and lse
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=q.device)

        num_segments = qo_indptr.numel() - 1
        for b in range(num_segments):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            Nq = q_end - q_start
            Nk = kv_end - kv_start

            if Nq <= 0 or Nk <= 0:
                continue

            delta = Nk - Nq

            # Launch Triton kernel: one program per (i, h)
            grid = (Nq * 32,)
            segment_attention_forward_kernel[grid](
                q_f32, k_f32, v_f32, output, lse,
                q_start, q_end, kv_start, kv_end, sm_scale, delta,
                num_warps=1, num_stages=1,
            )

        # Return outputs in original dtype and lse in float32 (base-2)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
