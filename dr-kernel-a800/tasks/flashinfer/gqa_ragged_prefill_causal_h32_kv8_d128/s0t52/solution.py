import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_all_segments_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,  # sm_scale is float32 scalar
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: process all query segments. One program per (i, h).
    - q_ptr: float32 [total_q, 32, 128]
    - k_ptr: float32 [total_kv, 8, 128]
    - v_ptr: float32 [total_kv, 8, 128]
    - out_ptr: float32 [total_q, 32, 128] (accumulator, will be zero-initialized on host)
    - lse_ptr: float32 [total_q, 32] (base-2 logsumexp, will be zero-initialized on host)
    - qo_indptr_ptr, kv_indptr_ptr: int32 [NUM_SEGMENTS+1]
    """
    i = tl.program_id(axis=0) // 32
    h = tl.program_id(axis=0) % 32

    # Process each segment b statically
    for b in tl.static_range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)      # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)    # int32
        kv_start = tl.load(kv_indptr_ptr + b)     # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1)   # int32

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq == 0 or Nk == 0:
            continue

        delta = Nk - Nq  # per-segment delta

        # Prepare logits for j in 0..7
        logits = tl.zeros((8,), dtype=tl.float32)

        # Load q[i, h, :]
        q_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_base)  # [128] float32

        # Compute dot-products with k's 8 heads, expanded to 32 query heads implicitly in accumulation
        # For each row (key token), accumulate dot(q_vec, k_row_j) * sm_scale into logits[j]
        for row in tl.static_range(Nk):
            for j in tl.static_range(8):
                orig_h = h % 8  # since k has 8 heads; for each j, we use head j
                k_off = kv_start + row
                k_vec = tl.load(k_ptr + k_off * 8 * 128 + j * 128 + orig_h * 128)  # [128] float32
                logits[j] += tl.sum(q_vec * k_vec) * sm_scale

        # Apply forward-causal mask: j >= (i + 1 + delta) -> -inf
        for j in tl.static_range(8):
            cond = (i + 1 + delta) <= j
            logits[j] = tl.where(cond, -float("inf"), logits[j])

        # Compute base-2 logsumexp over 8 positions
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions
        soft = tl.exp(logits - lse_val)  # [8], float32

        # Accumulate output[i, h, :] += soft[j] * v[kv_start + row, j, orig_h] for all rows
        for row in tl.static_range(Nk):
            for j in tl.static_range(8):
                orig_h = h % 8
                v_off = kv_start + row
                v_vec = tl.load(v_ptr + v_off * 8 * 128 + j * 128 + orig_h * 128)  # [128] float32
                out_base = (q_start + i) * 32 * 128 + h * 128
                tl.atomic_add(out_ptr + out_base, v_vec * soft[j])

        # Store lse[i, h]
        lse_idx = (q_start + i) * 32 + h
        tl.store(lse_ptr + lse_idx, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward: all heavy computation inside Triton kernel.
        Returns: output [total_q, 32, 128] bfloat16, lse [total_q, 32] float32 (base-2 logsumexp)
        """
        # Cast to float32 for compute; make contiguous
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        k_f32 = k.to(torch.float32).contiguous()  # [total_kv, 8, 128]
        v_f32 = v.to(torch.float32).contiguous()  # [total_kv, 8, 128]

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]

        # Output and lse buffers
        output = torch.zeros(
            (total_q, 32, 128), dtype=torch.float32, device=q.device
        )
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=q.device)

        NUM_SEGMENTS = qo_indptr.numel() - 1  # number of segments

        # Launch Triton kernel: one program per (i, h)
        grid = (total_q * 32,)
        attention_gqa_all_segments_kernel[grid](
            q_f32, k_f32, v_f32, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Cast output to bfloat16 to match original; keep lse float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
