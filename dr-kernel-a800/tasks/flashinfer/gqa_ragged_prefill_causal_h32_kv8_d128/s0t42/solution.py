import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_row_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: 2D grid over (segment b, query position i).
    For each (b, i), computes attention across q heads h and kv positions j=0..7,
    applies forward-causal mask, computes lse, softmax, and atomically accumulates
    into output[i, h, :] and stores lse[i, h].
    - q_ptr: float32 [total_q, 32, 128]
    - k_ptr: float32 [total_kv, 8, 128] (we map q head to kv head implicitly via modulo)
    - v_ptr: float32 [total_kv, 32, 128] (pre-expanded from original 8 heads)
    - out_ptr: float32 [total_q, 32, 128], initialized to zeros
    - lse_ptr: float32 [total_q, 32], initialized to -inf
    - qo_indptr_ptr, kv_indptr_ptr: int32 [NUM_SEGMENTS+1]
    """
    b = tl.program_id(axis=0)
    i = tl.program_id(axis=1)  # query position in this segment

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b)      # int32
    q_end = tl.load(qo_indptr_ptr + b + 1)    # int32
    kv_start = tl.load(kv_indptr_ptr + b)     # int32
    kv_end = tl.load(kv_indptr_ptr + b + 1)   # int32

    Nq = q_end - q_start
    Nk = kv_end - kv_start
    if i >= Nq:
        return

    # delta for this segment
    delta = Nk - Nq

    # Compute q_vec = q[q_start + i, h, :] for all h; we'll load per h
    # Prepare logits vector for 8 positions
    logits = tl.zeros((8,), dtype=tl.float32)

    # For each q head h
    for h in tl.static_range(32):
        # Load q[i, h, :]
        q_row_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_row_base)  # [128] float32

        # Compute dot with each kv head j=0..7 mapped to kv orig_h = h % 8
        for j in tl.static_range(8):
            orig_h = h % 8
            k_base = (kv_start + j) * 8 * 128 + orig_h * 128  # k has 8 heads
            k_vec = tl.load(k_ptr + k_base)  # [128]
            dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
            logits[j] = dot
            # Apply forward-causal mask: if j >= (i + 1 + delta), set -inf
            if j >= (i + 1 + delta):
                logits[j] = -float("inf")

        # Base-2 logsumexp over 8 positions
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions
        soft = tl.exp(logits - lse_val)  # [8], float32

        # Accumulate output[i, h, :] = sum over j of soft[j] * v[kv_start + j, orig_h, :]
        out_row_base = (q_start + i) * 32 * 128 + h * 128
        for j in tl.static_range(8):
            orig_h = h % 8
            v_base = (kv_start + j) * 8 * 128 + orig_h * 128
            v_vec = tl.load(v_ptr + v_base)  # [128]
            # Atomic add because multiple segments may contribute to the same (i, h)
            tl.atomic_add(out_ptr + out_row_base, v_vec * soft[j])

        # Store lse[i, h]
        lse_index = (q_start + i) * 32 + h
        tl.store(lse_ptr + lse_index, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward:
        - q: [total_q, 32, 128], bfloat16
        - k: [total_kv, 8, 128], bfloat16
        - v: [total_kv, 8, 128], bfloat16
        - qo_indptr: int32 [len_indptr], segments of queries
        - kv_indptr: int32 [len_indptr], segments of keys/values
        - sm_scale: float32 (e.g., 1/sqrt(128))
        Returns:
        - output: [total_q, 32, 128], float32 (we'll cast to bfloat16 at end)
        - lse: [total_q, 32], float32 (base-2 logsumexp)
        """
        device = q.device

        # Cast inputs to float32 for Triton computations
        q_f32 = q.contiguous().to(torch.float32)
        # Expand k and v to 32 heads to match q heads (GQA ratio = 4)
        # k_exp: [total_kv, 32, 128]
        k_exp = k.repeat_interleave(4, dim=1).contiguous().to(torch.float32)
        v_exp = v.repeat_interleave(4, dim=1).contiguous().to(torch.float32)

        total_q = q_f32.shape[0]
        total_kv = k_exp.shape[0]

        # Allocate outputs (float32)
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        NUM_SEGMENTS = qo_indptr.numel() - 1
        grid = (NUM_SEGMENTS, total_q)

        segment_attention_row_kernel[grid](
            q_f32, k_exp, v_exp, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
