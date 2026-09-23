import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_one_ih_kernel(
    q_ptr, k_ptr, v_exp_ptr, out_ptr, lse_ptr,
    q_start, q_end, kv_start, kv_end, sm_scale, delta,
    NUM_Q, NUM_K,
):
    """
    Triton kernel: process one (i, h). Assumes grid size is NUM_Q * 32 (one program per (i,h)).
    q_ptr: float32 [NUM_Q, 32, 128]
    k_ptr: float32 [NUM_K, 8, 128]
    v_exp_ptr: float32 [NUM_K, 32, 128]  (v expanded to 32 heads in host)
    out_ptr: float32 [NUM_Q, 32, 128]
    lse_ptr: float32 [NUM_Q, 32] (base-2 logsumexp)
    """
    # Each program handles one (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    if i >= NUM_Q:
        return

    # Load q[i, h, :] vector
    q_lin = (q_start + i) * 32 * 128 + h * 128
    q_vec = tl.load(q_ptr + q_lin)  # [128] float32

    # Prepare vectors for logits and lse
    logits = tl.zeros((8,), dtype=tl.float32)  # per j in [0..7]
    out_vec = tl.zeros((128,), dtype=tl.float32)  # output for this (i,h)

    # Compute dot-products for j in 0..7 and apply causal mask
    for j in tl.static_range(8):
        orig_h = h % 8  # map q head to kv group head
        k_lin = (kv_start + j) * 8 * 128 + orig_h * 128
        k_vec = tl.load(k_ptr + k_lin)  # [128] float32
        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar
        # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
        if j >= (i + 1 + delta):
            dot = -float("inf")
        # Store dot into logits vector
        logits = tl.where(tl.arange(0, 8) == j, dot, logits)

    # Compute base-2 logsumexp over 8 positions
    m = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - m), axis=0)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1/ln(2)

    # Softmax across j
    soft = tl.exp(logits - lse_val)  # [8] float32

    # Accumulate output[i, h, :] = sum_j soft[j] * v_exp[kv_start + j, orig_h, :]
    for j in tl.static_range(8):
        orig_h = h % 8
        v_lin = (kv_start + j) * 32 * 128 + orig_h * 128
        v_vec = tl.load(v_exp_ptr + v_lin)  # [128] float32
        out_vec += soft[j] * v_vec

    # Store output[i, h, :]
    out_lin = i * 32 * 128 + h * 128
    tl.store(out_ptr + out_lin, out_vec)

    # Store lse[i, h] (base-2)
    lse_lin = i * 32 + h
    tl.store(lse_ptr + lse_lin, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # q: [total_q, 32, 128], bfloat16
        # k: [total_kv, 8, 128], bfloat16
        # v: [total_kv, 8, 128], bfloat16
        # qo_indptr: int32 [len_indptr]
        # kv_indptr: int32 [len_indptr]
        # sm_scale: float32

        # Assertions consistent with the original code
        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8
        assert q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128
        # We'll process all segments in host, launch kernel per segment

        total_q = int(q.shape[0])
        total_kv = int(k.shape[0])
        num_qo_heads = 32
        num_kv_heads = 8

        # Output and lse tensors
        output = torch.zeros((total_q, num_qo_heads, 128), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

        # Precompute expanded v to 32 heads (GQA mapping 8->32 by repeat_interleave)
        # This mimics v_expanded in the original code
        v_exp = v.repeat_interleave(4, dim=1).to(torch.float32).contiguous()

        # Iterate over segments b
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No queries or KV for this batch element
                continue

            Nq = q_end - q_start
            Nk = kv_end - kv_start

            # Cast q and k to float32 for computation
            q_f32 = q[q_start:q_end].to(torch.float32).contiguous()  # [Nq, 32, 128]
            k_f32 = k[kv_start:kv_end].to(torch.float32).contiguous()  # [Nk, 8, 128]

            # Launch one program per (i, h)
            grid = (Nq * num_qo_heads,)
            attention_forward_one_ih_kernel[grid](
                q_f32, k_f32, v_exp, output, lse,
                q_start, q_end, kv_start, kv_end, sm_scale, (Nk - Nq),
                Nq, Nk,
                num_warps=1, num_stages=1,
            )

        # Return output in original dtype and lse in float32 (base-2)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
