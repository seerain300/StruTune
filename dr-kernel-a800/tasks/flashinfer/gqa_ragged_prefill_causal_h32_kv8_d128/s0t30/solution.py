import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_lse_kernel(
    q_ptr, k_ptr, out_logits_ptr, lse_ptr,
    q_start, q_end, kv_start, kv_end, sm_scale,
    NUM_Q, NUM_K, delta,
):
    """
    Compute logits[i, h, j] for all i in [0, NUM_Q), h in [0, 31], j in 0..7.
    Store lse[i, h] base-2. Each program handles one (i, h).
    q_ptr: [NUM_Q, 32, 128] float32
    k_ptr: [NUM_K, 8, 128] float32
    out_logits_ptr: [NUM_Q, 32, 8] float32
    lse_ptr: [NUM_Q, 32] float32
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    if i >= NUM_Q:
        return

    # Load q[i, h, :]
    q_lin = (q_start + i) * 32 * 128 + h * 128
    q_vec = tl.load(q_ptr + q_lin)  # [128] float32

    # Prepare vector for logsumexp across j
    logits_vec = tl.zeros((8,), dtype=tl.float32)

    # Compute dot-products for j in 0..7 and apply causal mask
    for j in tl.static_range(8):
        orig_h = h % 8  # map q head to kv group head
        k_lin = (kv_start + j) * 8 * 128 + orig_h * 128
        k_vec = tl.load(k_ptr + k_lin)  # [128] float32

        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        if j >= (i + 1 + delta):
            dot = -float("inf")
        # Store dot into out_logits[i, h, j]
        out_lin = i * 32 * 8 + h * 8 + j
        tl.store(out_logits_ptr + out_lin, dot)

        # Accumulate for logsumexp
        logits_vec[j] = dot

    # Base-2 logsumexp
    m = tl.max(logits_vec, axis=0)
    sum_exp = tl.sum(tl.exp(logits_vec - m), axis=0)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)
    lse_lin = i * 32 + h
    tl.store(lse_ptr + lse_lin, lse_val)


@triton.jit
def compute_output_kernel(
    lse_ptr, out_logits_ptr, v_exp_ptr, out_ptr,
    q_start, q_end, kv_start, kv_end,
    NUM_Q, NUM_K, delta,
):
    """
    Compute output[i, h, :] for all i in [0, NUM_Q), h in [0, 31].
    Uses lse[i, h] and logits[i, h, :] to compute softmax over j (8) and then
    out[i, h, :] = sum_j softmax[j] * v_exp[kv_start + j, orig_h, :], where v_exp is [NUM_K, 32, 128].
    Each program handles one (i, h).
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    if i >= NUM_Q:
        return

    # Load lse[i, h]
    lse_lin = i * 32 + h
    lse_val = tl.load(lse_ptr + lse_lin)

    # Prepare output vector [128]
    out_vec = tl.zeros((128,), dtype=tl.float32)

    # For each j, compute softmax[j] and weighted v_exp row
    for j in tl.static_range(8):
        orig_h = h % 8
        # Load logits[i, h, j]
        out_lin = i * 32 * 8 + h * 8 + j
        logits_ij = tl.load(out_logits_ptr + out_lin)

        # Softmax across j: P(j) = exp(logits_ij - lse_val)
        soft_j = tl.exp(logits_ij - lse_val)

        # Load v_exp[kv_start + j, orig_h, :] which is [128]
        v_lin = (kv_start + j) * 32 * 128 + orig_h * 128
        v_vec = tl.load(v_exp_ptr + v_lin)  # [128]
        out_vec += soft_j * v_vec

    # Store out[i, h, :]
    out_lin_base = (q_start + i) * 32 * 128 + h * 128
    tl.store(out_ptr + out_lin_base, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128] bfloat16
        k: [total_kv, 8, 128] bfloat16
        v: [total_kv, 8, 128] bfloat16
        qo_indptr: [len_indptr] int32
        kv_indptr: [len_indptr] int32
        sm_scale: float32
        Returns:
        - output: [total_q, 32, 128] float32
        - lse: [total_q, 32] float32 (base-2 logsumexp)
        """
        device = q.device
        total_q = q.shape[0]
        total_kv = k.shape[0]

        # Materialize expanded k and v (GQA: 8->32)
        k_exp = k.to(torch.float32).repeat_interleave(4, dim=1).contiguous()  # [total_kv, 32, 128]
        v_exp = v.to(torch.float32).repeat_interleave(4, dim=1).contiguous()  # [total_kv, 32, 128]

        # Allocate outputs
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        # Determine number of segments
        NUM_SEGMENTS = qo_indptr.numel() - 1

        # Loop over segments; launch Triton kernels per segment
        for b in range(NUM_SEGMENTS):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            NUM_Q = q_end - q_start
            NUM_K = kv_end - kv_start
            if NUM_Q == 0 or NUM_K == 0:
                continue

            delta = NUM_K - NUM_Q

            # Temporary buffers for logits and lse
            out_logits = torch.empty((NUM_Q, 32, 8), dtype=torch.float32, device=device)
            lse_buf = torch.empty((NUM_Q, 32), dtype=torch.float32, device=device)

            # Grid spans total_q * 32 programs, each computes (i, h)
            grid = (NUM_Q * 32,)

            # Compute logits and lse
            compute_logits_lse_kernel[grid](
                q.to(torch.float32), k.to(torch.float32), out_logits, lse_buf,
                q_start, q_end, kv_start, kv_end, sm_scale,
                NUM_Q, NUM_K, delta,
                num_warps=4, num_stages=2
            )

            # Compute output using softmax and v_exp
            compute_output_kernel[grid](
                lse_buf, out_logits, v_exp, output,
                q_start, q_end, kv_start, kv_end,
                NUM_Q, NUM_K, delta,
                num_warps=4, num_stages=2
            )

        return output, lse


# Optional: Provide Model entry point if evaluator expects it.
class Model(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        return ModelNew()(q, k, v, qo_indptr, kv_indptr, sm_scale)


def run(*args):
    return ModelNew()(*args)
