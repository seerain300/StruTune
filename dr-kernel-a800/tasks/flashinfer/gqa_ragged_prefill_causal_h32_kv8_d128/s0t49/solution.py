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
    Triton kernel: process one segment b per program.
    q_ptr: float32 [total_q, 32, 128]
    k_exp_ptr: float32 [total_kv, 32, 128] (pre-expanded keys)
    v_exp_ptr: float32 [total_kv, 32, 128] (pre-expanded values)
    out_ptr: float32 [total_q, 32, 128], initialized by host
    lse_ptr: float32 [total_q, 32], initialized by host
    qo_indptr_ptr, kv_indptr_ptr: int32 [len_indptr+1]
    total_q, total_kv, sm_scale: scalars
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

    # segment-level delta for forward-causal mask
    delta = Nk - Nq  # int32 scalar

    # Iterate over query positions within this segment
    i = 0
    while i < Nq:
        # Compute and store output and lse per query head h
        for h in tl.static_range(0, 32):
            # We compute logits for j in 0..7 (8 positions) and then softmax and accumulation
            # Initialize logits vector for 8 positions
            logits = tl.zeros((8,), dtype=tl.float32)

            for j in tl.static_range(0, 8):
                # Load q[i, h, :] vector (128 elements)
                q_base = (q_start + i) * 32 * 128 + h * 128
                q_vec = tl.load(q_ptr + q_base)  # [128] float32

                # Load k_exp[kv_start + t, j, :] for t in [kv_start, kv_end)
                # We need the sum over t of q_vec * k_exp[t, j, :]
                # Triton supports pointer arithmetic; we'll compute dot as a scalar.
                # Initialize dot accumulator
                dot_acc = tl.zeros((), dtype=tl.float32)

                t = kv_start
                while t < kv_end:
                    k_base = t * 32 * 128 + j * 128
                    k_line = tl.load(k_exp_ptr + k_base)  # [128] float32
                    dot_acc += tl.sum(q_vec * k_line)  # scalar float32
                    t += 1

                dot_acc = dot_acc * sm_scale

                # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
                if (j >= (i + 1 + delta)):
                    logits[j] = -float('inf')
                else:
                    logits[j] = dot_acc

            # Compute base-2 logsumexp over 8 positions
            m = tl.max(logits, axis=0)
            sum_exp = tl.sum(tl.exp(logits - m), axis=0)
            lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

            # Softmax across the 8 positions
            soft = tl.exp(logits - lse_val)  # [8] float32

            # Accumulate output[i, h, :] += soft[j] * v_exp[kv_start + t, j, :] for each t
            # Since soft is over the 8 positions (j), and we've already accumulated into out[i,h,:]
            # We need to compute and store the final output vector for this (i,h)
            out_base = (q_start + i) * 32 * 128 + h * 128
            out_vec = tl.load(out_ptr + out_base)  # [128] float32 (initialize from host zeros)
            # For each j, add contribution soft[j] * v_exp[kv_start + t, j, :] over all t in [kv_start, kv_end)
            # We iterate j from 0 to 7; for each j, iterate t from kv_start to kv_end, load v_exp[t, j, :] and add
            for jj in tl.static_range(0, 8):
                contrib_acc = tl.zeros((128,), dtype=tl.float32)
                t = kv_start
                while t < kv_end:
                    v_base = t * 32 * 128 + jj * 128
                    v_line = tl.load(v_exp_ptr + v_base)  # [128] float32
                    contrib_acc += soft[jj] * v_line
                    t += 1

                out_vec += contrib_acc

            # Store final output vector for this (i,h)
            tl.store(out_ptr + out_base, out_vec)

            # Store lse[i, h] (base-2)
            lse_lin = (q_start + i) * 32 + h
            tl.store(lse_ptr + lse_lin, lse_val)

        i += 1

# Optional helper to pre-expand k and v to 32 heads
def _repeat_interleave_dim1_4(x: torch.Tensor) -> torch.Tensor:
    # x is [N, 8, 128] or [N, 32, 128] depending; here we expand 8->32
    # Use expand + contiguous: repeat_interleave is data movement
    # We can simply do .repeat_interleave(4, dim=1)
    return x.repeat_interleave(4, dim=1)

class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        device = q.device

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Pre-expand k and v to 32 heads (GQA mapping 8->32 via repeat_interleave)
        k_exp = _repeat_interleave_dim1_4(k_f32)
        v_exp = _repeat_interleave_dim1_4(v_f32)

        # Prepare output and lse tensors (float32 for computation)
        total_q = q_f32.shape[0]
        total_kv = k_exp.shape[0]
        num_qo_heads = q_f32.shape[1]
        out = torch.zeros((total_q, num_qo_heads, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)

        # Number of segments
        NUM_SEGMENTS = qo_indptr.numel() - 1
        grid = (NUM_SEGMENTS,)
        segment_attention_gqa_kernel[grid](
            q_f32, k_exp, v_exp, out, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
        )

        # Cast output back to bfloat16 to match original return type expectation
        output_bf16 = out.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
