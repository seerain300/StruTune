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
    Pre-expanded k and v are provided:
      k_exp_ptr: float32 [total_kv, 32, 128]
      v_exp_ptr: float32 [total_kv, 32, 128]
    Outputs:
      out_ptr: float32 [total_q, 32, 128], will be written by kernel
      lse_ptr: float32 [total_q, 32], will be written by kernel
    Segment bounds are read from qo_indptr_ptr and kv_indptr_ptr.
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
        # For each query head h
        for h in tl.static_range(0, 32):
            # We will compute logits for j in 0..7
            logits = tl.zeros((8,), dtype=tl.float32)
            for j in tl.static_range(0, 8):
                # Load q[i, h, :]
                q_base = (q_start + i) * 32 * 128 + h * 128
                q_vec = tl.load(q_ptr + q_base)  # [128] float32

                # Accumulate dot = sum(q_vec * k_exp[t, j, :]) over t in [kv_start, kv_end)
                dot_acc = tl.zeros((), dtype=tl.float32)
                t = kv_start
                while t < kv_end:
                    k_base = t * 32 * 128 + j * 128  # k_exp[t, j, :]
                    k_vec = tl.load(k_exp_ptr + k_base)  # [128] float32
                    # Multiply and sum across 128 dims
                    dot_acc += tl.sum(q_vec * k_vec, axis=0)
                    t += 1
                # Scale
                dot_acc *= sm_scale
                # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
                if j >= (i + 1 + delta):
                    dot_acc = -float("inf")
                logits[j] = dot_acc

            # lse in base-2
            m = tl.max(logits, axis=0)
            sum_exp = tl.sum(tl.exp(logits - m), axis=0)
            lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1/ln(2)

            # Softmax over 8 positions
            sum_exp_soft = tl.sum(tl.exp(logits - lse_val), axis=0)
            soft = tl.exp(logits - lse_val)  # [8]

            # Accumulate output[i, h, :] += soft[j] * v_exp[t, j, :] for t in [kv_start, kv_end)
            out_base = (q_start + i) * 32 * 128 + h * 128
            # Initialize output vector to zeros
            out_vec = tl.zeros((128,), dtype=tl.float32)
            t = kv_start
            while t < kv_end:
                for j in tl.static_range(0, 8):
                    v_base = t * 32 * 128 + j * 128  # v_exp[t, j, :]
                    v_vec = tl.load(v_exp_ptr + v_base)  # [128]
                    out_vec += v_vec * soft[j]
                t += 1
            # Store output
            tl.store(out_ptr + out_base, out_vec)

            # Store lse[i, h]
            lse_lin = (q_start + i) * 32 + h
            tl.store(lse_ptr + lse_lin, lse_val)

        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward that replaces all torch computations with Triton kernels.
        - q: [total_q, 32, 128], bfloat16
        - k: [total_kv, 8, 128], bfloat16
        - v: [total_kv, 8, 128], bfloat16
        - qo_indptr: int32 [len_indptr+1]
        - kv_indptr: int32 [len_indptr+1]
        - sm_scale: float32
        Returns:
        - output: [total_q, 32, 128], float32
        - lse: [total_q, 32], float32 (base-2 logsumexp)
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors."
        device = q.device
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.numel() - 1

        # Expand k and v to 32 heads (GQA ratio)
        k_exp = k.repeat_interleave(4, dim=1).to(torch.float32).contiguous()
        v_exp = v.repeat_interleave(4, dim=1).to(torch.float32).contiguous()

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per segment
        NUM_SEGMENTS = len_indptr
        grid = (NUM_SEGMENTS,)
        segment_attention_gqa_kernel[grid](
            q.to(torch.float32).contiguous(), k_exp, v_exp, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
