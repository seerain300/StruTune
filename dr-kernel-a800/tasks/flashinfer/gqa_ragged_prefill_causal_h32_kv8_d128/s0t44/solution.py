import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_segments_kernel(
    q_ptr, k_exp_ptr, v_exp_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
):
    """
    Triton kernel: 2D grid over (segment b, query position i).
    For each (b, i), compute attention for all q heads h:
      - Load q[i, h, :] (float32).
      - For j in 0..7, compute dot(q, k_exp[j, h%8, :]) and apply causal mask per segment: if j >= (i + 1 + delta), set -inf.
      - Compute base-2 logsumexp over 8 positions, softmax, and accumulate output[i, h, :] += soft[j] * v_exp[j, h%8, :].
      - Atomically add to global out_ptr and lse_ptr.
    """
    b = tl.program_id(axis=0)
    i = tl.program_id(axis=1)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b)   # int32
    q_end = tl.load(qo_indptr_ptr + b + 1)  # int32
    kv_start = tl.load(kv_indptr_ptr + b)   # int32
    kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

    Nq = q_end - q_start
    Nk = kv_end - kv_start

    if Nq == 0 or Nk == 0:
        return

    delta = Nk - Nq

    # Process all 32 query heads
    for h in tl.static_range(32):
        # Base for q[i, h, :]
        q_row_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_row_base)  # [128] float32

        # Prepare logits (length 8)
        logits = tl.zeros((8,), dtype=tl.float32)

        # Compute dot products with each kv head j=0..7, map q head to kv head orig_h = h % 8
        for j in tl.static_range(8):
            orig_h = h % 8
            # Load k_exp[j, orig_h, :]
            k_base = (kv_start + j) * 8 * 128 + orig_h * 128
            k_vec = tl.load(k_exp_ptr + k_base)  # [128]
            dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar
            # Apply forward-causal mask: if j >= (i + 1 + delta), set -inf
            if j >= (i + 1 + delta):
                dot = -float("inf")
            logits[j] = dot

        # Base-2 logsumexp across 8 positions
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions (broadcast to 128)
        soft = tl.exp(logits - lse_val)  # [8] float32

        # Accumulate output[i, h, :] += soft[j] * v_exp[j, orig_h, :] for j in 0..7
        out_row_base = (q_start + i) * 32 * 128 + h * 128
        # Atomic add to avoid overwrites across segments
        # Note: out_ptr is float32, initialize zeros on host
        for j in tl.static_range(8):
            orig_h = h % 8
            v_base = (kv_start + j) * 8 * 128 + orig_h * 128
            v_vec = tl.load(v_exp_ptr + v_base)  # [128]
            contrib = v_vec * soft[j]  # [128]
            tl.atomic_add(out_ptr + out_row_base, contrib)

        # Atomic add lse[i, h]
        lse_index = (q_start + i) * 32 + h
        tl.atomic_add(lse_ptr + lse_index, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward. All computation inside Triton kernels.
        Returns:
          - output: [total_q, 32, 128] float32
          - lse: [total_q, 32] float32 (base-2 logsumexp)
        """
        # Ensure inputs are contiguous and float32 for Triton
        q_f32 = q.to(torch.float32).contiguous()
        # Create expanded k and v to 32 heads (repeat_interleave 4 times)
        # Note: reshape to [Nk, 8, 128] then repeat
        # We need total_kv from kv_indptr[-1] to compute Nk per segment. Easiest is to build k_exp and v_exp for each segment in Triton by indexing, but Triton kernel here accesses directly k/v.
        # For Triton kernel to work, we need k_expanded and v_expanded as separate tensors. We will create them using PyTorch and pass to kernel.
        # We must compute q_end and kv_end for each b; since kernel iterates over b, we can pass tensors directly.
        # However, Triton cannot index 2D tensors with dynamic bases in complex ways; better to precompute q_end and kv_end vectors on host and pass.
        # But Triton kernel only needs qo_indptr and kv_indptr; Nq and Nk are computed inside. We can pass qo_indptr and kv_indptr; Nq/Nk are derived inside.

        # Prepare output and lse tensors (float32)
        total_q = q_f32.shape[0]
        total_kv = kv_indptr[-1].item()
        out = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=q.device)

        # Launch Triton kernel over 2D grid: (segments, query positions)
        NUM_SEGMENTS = qo_indptr.numel() - 1
        grid = (NUM_SEGMENTS, total_q)
        attention_gqa_segments_kernel[grid](
            q_f32, k.to(torch.float32).contiguous(), v.to(torch.float32).contiguous(),
            out, lse, qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            num_warps=4, num_stages=2
        )

        # Return output in bfloat16 to match original example outputs; lse in float32
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
