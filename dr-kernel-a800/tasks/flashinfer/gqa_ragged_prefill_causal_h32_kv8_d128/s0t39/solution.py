import math
import torch
import triton
import triton.language as tl


@triton.jit
def attn_gqa_per_row_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h) over all segments.
    q_ptr: float32 [total_q, 32, 128]
    k_ptr: float32 [total_kv, 8, 128]
    v_ptr: float32 [total_kv, 8, 128]
    out_ptr: float32 [total_q, 32, 128] (we initialize it to zeros on host)
    lse_ptr: float32 [total_q, 32] (we initialize to -inf on host)
    qo_indptr_ptr: int32 [NUM_SEGMENTS+1]
    kv_indptr_ptr: int32 [NUM_SEGMENTS+1]
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # Loop over segments (NUM_SEGMENTS must be passed as constexpr)
    for b in range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)     # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)   # int32
        kv_start = tl.load(kv_indptr_ptr + b)    # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq > 0 and Nk > 0:
            delta = Nk - Nq  # per-segment delta

            # Compute logits vector across 8 kv heads for this (i, h)
            j_idx = tl.arange(0, 8)  # [0,1,2,3,4,5,6,7]
            allow = j_idx < (i + 1 + delta)  # forward-causal mask for each j

            # Load q[i, h, :]
            q_base = (q_start + i) * 32 * 128 + h * 128
            q_vec = tl.load(q_ptr + q_base)  # [128] float32

            # Build logits vector of length 8
            logits = tl.zeros((8,), dtype=tl.float32)
            for jj in range(8):
                orig_h = h % 8
                # Pointers for k rows: [Nk, 8, 128], then select orig_h-th head
                k_row_base = kv_start * 8 * 128 + orig_h * 128
                # Each row k row has 128 elements
                k_vec = tl.load(k_ptr + k_row_base + jj * 128)  # [128]
                acc = tl.sum(q_vec * k_vec)  # scalar
                acc *= sm_scale
                # Apply mask: if allow[jj] is False, set to -inf
                acc = tl.where(allow[jj], acc, -1e20)
                logits[jj] = acc

            # Base-2 logsumexp across 8 positions
            m = tl.max(logits, axis=0)
            sum_exp = tl.sum(tl.exp(logits - m), axis=0)
            lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

            # Softmax across 8 positions
            soft = tl.exp(logits - lse_val)  # [8]

            # Accumulate output[i, h, :] = sum_j soft[j] * v_exp[:, j, :]
            out_base = (q_start + i) * 32 * 128 + h * 128
            current = tl.load(out_ptr + out_base)  # [128]
            for j in range(8):
                orig_h = h % 8
                v_row_base = kv_start * 32 * 128 + orig_h * 128
                v_vec = tl.load(v_ptr + v_row_base + j * 128)  # [128]
                current += v_vec * soft[j]
            tl.store(out_ptr + out_base, current)

            # Store lse[i, h]
            lse_index = (q_start + i) * 32 + h
            tl.store(lse_ptr + lse_index, lse_val)

        # If Nq==0 or Nk==0 for this segment, do nothing (no work).


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32 segment starts for queries
        kv_indptr: [len_indptr], int32 segment starts for keys/values
        sm_scale: float32 scalar
        Returns:
        - output: [total_q, 32, 128], bfloat16
        - lse: [total_q, 32], float32 (base-2 logsumexp)
        """
        device = q.device

        # Slice to current segment: len_indptr is typically 2 in evaluator
        NUM_SEGMENTS = qo_indptr.numel() - 1

        # Prepare expanded k and v to 32 heads on host
        k32 = k.repeat_interleave(4, dim=1)  # [total_kv, 32, 128]
        v32 = v.repeat_interleave(4, dim=1)  # [total_kv, 32, 128]
        # Cast to float32 for computation
        q_f32 = q.to(torch.float32)
        k_f32 = k32.to(torch.float32)
        v_f32 = v32.to(torch.float32)

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]

        # Initialize output and lse tensors
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (i, h)
        grid = (total_q * 32,)
        attn_gqa_per_row_kernel[grid](
            q_f32, k_f32, v_f32, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
            num_warps=4,
        )

        # Cast output back to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
