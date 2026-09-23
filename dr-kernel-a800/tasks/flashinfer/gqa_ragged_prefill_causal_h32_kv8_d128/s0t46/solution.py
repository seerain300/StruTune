import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_gqa_kernel(
    q_ptr, k_exp_ptr, v_exp_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
):
    """
    Triton kernel: one program per segment b.
    - q_ptr: float32 [total_q, 32, 128]
    - k_exp_ptr: float32 [total_kv, 32, 128] (pre-expanded k to 32 heads)
    - v_exp_ptr: float32 [total_kv, 32, 128] (pre-expanded v to 32 heads)
    - out_ptr: float32 [total_q, 32, 128], initialized by host
    - lse_ptr: float32 [total_q, 32], initialized by host
    - qo_indptr_ptr: int32 [len_indptr+1]
    - kv_indptr_ptr: int32 [len_indptr+1]
    - total_q, total_kv, sm_scale
    """
    b = tl.program_id(axis=0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b)
    q_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    Nq = q_end - q_start
    Nk = kv_end - kv_start

    if Nq == 0 or Nk == 0:
        return

    delta = Nk - Nq  # segment-level delta

    # Iterate over query positions within this segment
    i = 0
    while i < Nq:
        # For each query head h, compute logits for j in 0..7, then lse and output
        for h in tl.static_range(0, 32):
            # Prepare logits vector for j in 0..7
            logits = tl.zeros((8,), dtype=tl.float32)

            # Compute dot products for j in 0..7
            for j in tl.static_range(0, 8):
                # Load q[i, h, :]
                q_base = (q_start + i) * 32 * 128 + h * 128
                q_vec = tl.load(q_ptr + q_base)  # [128] float32

                # Accumulate dot = sum(q_vec * k_exp[t, j, :]) over t in [kv_start, kv_end)
                dot_acc = tl.zeros((), dtype=tl.float32)
                t = kv_start
                while t < kv_end:
                    k_base = t * 32 * 128 + j * 128  # k_exp[t, j, :]
                    k_vec = tl.load(k_exp_ptr + k_base)  # [128]
                    dot_acc += tl.sum(q_vec * k_vec, axis=0)
                    t += 1

                # Apply causal mask: if j >= (i + 1 + delta), set to -inf
                if (j >= (i + 1 + delta)):
                    dot_acc = -1e20  # float32 -inf surrogate

                # Scale by sm_scale
                dot_acc *= sm_scale

                # Store in logits vector
                logits[j] = dot_acc

            # Compute base-2 logsumexp over the 8 positions
            m = tl.max(logits, axis=0)
            sum_exp = tl.sum(tl.exp(logits - m), axis=0)
            lse_val = m + tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2)

            # Softmax across 8 positions
            soft = tl.exp(logits - lse_val)  # [8], float32

            # Initialize output row
            out_base = (q_start + i) * 32 * 128 + h * 128
            out_row = tl.zeros((128,), dtype=tl.float32)

            # Accumulate output[i, h, :] += soft[j] * v_exp[t, j, :] over j and t
            t = kv_start
            while t < kv_end:
                for jj in tl.static_range(0, 8):
                    v_base = t * 32 * 128 + jj * 128  # v_exp[t, jj, :]
                    v_vec = tl.load(v_exp_ptr + v_base)  # [128]
                    out_row += soft[jj] * v_vec
                t += 1

            # Store output row
            tl.store(out_ptr + out_base, out_row)

            # Store lse[i, h]
            lse_index = (q_start + i) * 32 + h
            tl.store(lse_ptr + lse_index, lse_val)

        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."

        # Cast to float32 for computation
        q_f32 = q.contiguous().to(torch.float32)
        # Pre-expand k and v to 32 heads: repeat_interleave along head dimension
        k_exp = k.contiguous().repeat_interleave(4, dim=1).to(torch.float32)
        v_exp = v.contiguous().repeat_interleave(4, dim=1).to(torch.float32)

        total_q = q_f32.shape[0]
        total_kv = k_exp.shape[0]

        # Output and lse buffers (float32)
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=q.device)

        # Number of segments
        NUM_SEGMENTS = qo_indptr.numel() - 1

        # Launch Triton kernel: one program per segment
        grid = (NUM_SEGMENTS,)
        segment_attention_gqa_kernel[grid](
            q_f32, k_exp, v_exp, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
        )

        # Cast output to bfloat16 to match original return type expectations
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
