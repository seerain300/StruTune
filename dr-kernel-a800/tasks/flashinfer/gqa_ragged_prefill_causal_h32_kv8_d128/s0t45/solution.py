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
    - k_exp_ptr: float32 [total_kv, 32, 128] (pre-expanded k)
    - v_exp_ptr: float32 [total_kv, 32, 128] (pre-expanded v)
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
        # Prepare lse for this i across all heads; we'll compute per h
        # We'll compute output[i, h, :] as we go. lse[i, h] we store after computing softmax.
        for h in tl.static_range(0, 32):
            # Base offset for output[i, h, :]
            out_base = (q_start + i) * 32 * 128 + h * 128

            # Compute logits for j in 0..7
            logits = tl.zeros((8,), dtype=tl.float32)
            for j in tl.static_range(0, 8):
                # Load q[i, h, :]
                q_base = (q_start + i) * 32 * 128 + h * 128
                q_vec = tl.load(q_ptr + q_base)  # [128] float32

                # Accumulate dot = sum(q_vec * k_exp[t, j, :]) over t in [kv_start, kv_end)
                dot_acc = tl.zeros((), dtype=tl.float32)
                t = kv_start
                while t < kv_end:
                    k_base = t * 32 * 128 + j * 128
                    k_vec = tl.load(k_exp_ptr + k_base)  # [128] float32
                    dot_acc += tl.sum(q_vec * k_vec, axis=0)
                    t += 1

                # Scale
                dot_acc *= sm_scale
                # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
                if (j >= (i + 1 + delta)):
                    dot_acc = -float("inf")

                logits[j] = dot_acc

            # Compute base-2 logsumexp over 8 logits
            m = tl.max(logits, axis=0)  # scalar
            sum_exp = tl.sum(tl.exp(logits - m), axis=0)  # scalar
            lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

            # Softmax across 8 positions
            soft = tl.exp(logits - lse_val)  # [8] float32

            # Accumulate output[i, h, :] = sum_j soft[j] * sum_t v_exp[t, j, :]
            out_vec = tl.zeros((128,), dtype=tl.float32)
            for j in tl.static_range(0, 8):
                s_j = soft[j]  # scalar
                acc = tl.zeros((), dtype=tl.float32)
                t = kv_start
                while t < kv_end:
                    v_base = t * 32 * 128 + j * 128
                    v_vec = tl.load(v_exp_ptr + v_base)  # [128] float32
                    acc += tl.sum(v_vec, axis=0)  # scalar
                    t += 1
                out_vec += s_j * acc

            # Store output[i, h, :]
            tl.store(out_ptr + out_base, out_vec)

        # Store lse[i, h] for h in 0..31
        for h in tl.static_range(0, 32):
            lse_base = (q_start + i) * 32 + h
            tl.store(lse_ptr + lse_base, lse_val)

        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Cast and make contiguous
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]

        # Pre-expand k and v to 32 heads
        gqa_ratio = 4  # 32 // 8
        k_exp = k_f32.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [total_kv, 32, 128]
        v_exp = v_f32.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [total_kv, 32, 128]

        # Output and lse initialization
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        NUM_SEGMENTS = qo_indptr.numel() - 1
        grid = (NUM_SEGMENTS,)

        segment_attention_gqa_kernel[grid](
            q_f32, k_exp, v_exp, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            num_warps=4, num_stages=2,
        )

        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
