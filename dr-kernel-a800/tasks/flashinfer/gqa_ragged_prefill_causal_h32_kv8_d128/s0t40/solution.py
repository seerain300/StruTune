import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_single_pos_kernel(
    q_ptr, k_exp_ptr, v_exp_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h) across all segments.
    - q_ptr: float32 [total_q, 32, 128]
    - k_exp_ptr: float32 [total_kv, 32, 128] (k expanded to 32 heads)
    - v_exp_ptr: float32 [total_kv, 32, 128] (v expanded to 32 heads)
    - out_ptr: float32 [total_q, 32, 128], to be accumulated into
    - lse_ptr: float32 [total_q, 32], -inf initialized, to be filled
    - qo_indptr_ptr, kv_indptr_ptr: int32 [NUM_SEGMENTS+1]
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # Loop over segments (compile-time constant bound)
    for b in tl.static_range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)  # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)  # int32
        kv_start = tl.load(kv_indptr_ptr + b)  # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq <= 0 or Nk <= 0:
            continue

        delta = Nk - Nq

        # Load q_vec = q[i, h, :]
        q_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_base)  # float32 [128]

        # Compute logits vector for j in 0..7: logits[j] = dot(q_vec, k_exp[:, j, :])
        logits = tl.zeros((8,), dtype=tl.float32)
        for j in range(8):
            # Accumulate dot product over k rows
            acc = 0.0
            for t in range(Nk):
                k_base = (kv_start + t) * 32 * 128 + j * 128
                k_vec = tl.load(k_exp_ptr + k_base)  # float32 [128]
                acc += tl.sum(q_vec * k_vec)
            logits[j] = acc * sm_scale

        # Apply forward-causal mask for this segment
        for j in range(8):
            if (j >= (i + 1 + delta)):
                logits[j] = -float("inf")

        # Compute base-2 logsumexp over 8 positions
        m = logits[0]
        for jj in range(1, 8):
            m = tl.maximum(m, logits[jj])
        sum_exp = 0.0
        for jj in range(8):
            sum_exp += tl.exp(logits[jj] - m)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1/ln(2)

        # Softmax over the 8 positions
        soft = tl.exp(logits - lse_val)  # [8] float32

        # Accumulate output[i, h, :] += soft[j] * v_exp[:, j, :] for j in 0..7
        out_base = (i * 32 + h) * 128
        for j in range(8):
            v_base = j * 128
            v_vec = tl.load(v_exp_ptr + v_base)  # [128]
            tl.atomic_add(out_ptr + out_base, v_vec * soft[j])

        # Store lse[i, h]
        tl.store(lse_ptr + (i * 32 + h), lse_val)

    # Final store (only one after all segments handled; but out_ptr is already accumulated)
    pass


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr, kv_indptr: int32 [len_indptr]
        sm_scale: float32
        Returns:
        - output: [total_q, 32, 128], bfloat16
        - lse: [total_q, 32], float32 base-2 logsumexp
        """
        device = q.device
        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Expand k and v to 32 heads (GQA ratio = 4)
        # k_expanded: [total_kv, 32, 128], v_expanded: [total_kv, 32, 128]
        gqa_ratio = 4  # 32 // 8
        k_exp = k_f32.repeat_interleave(gqa_ratio, dim=1).contiguous()
        v_exp = v_f32.repeat_interleave(gqa_ratio, dim=1).contiguous()

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]
        NUM_SEGMENTS = qo_indptr.numel() - 1

        # Allocate output and lse
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (i, h)
        grid = (total_q * 32,)
        attention_gqa_single_pos_kernel[grid](
            q_f32, k_exp, v_exp, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Cast output to bfloat16 to match original output dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
