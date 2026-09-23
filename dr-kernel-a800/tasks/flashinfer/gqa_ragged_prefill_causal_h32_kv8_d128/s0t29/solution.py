import math
import torch
import triton
import triton.language as tl


@triton.jit
def expand_k_v_kernel(
    k_ptr, v_ptr, k_exp_ptr, v_exp_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,  # sm_scale unused
    NUM_GROUPS: tl.constexpr,     # num kv heads = 8
    REPEAT: tl.constexpr,         # expansion factor = 4
):
    """
    Triton kernel: for each segment b, and each original kv row r, expand 8 heads to 32 heads
    by repeating each head 4 times. We index k_ptr and v_ptr by (r, h, d), and write to
    k_exp_ptr and v_exp_ptr at (segment_index, expanded_h, d).
    """
    b = tl.program_id(axis=0)  # segment index
    r = tl.program_id(axis=1)  # original row index within this segment

    # Load segment starts
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Guard: if this program id maps out of segment bounds, return
    if kv_start >= kv_end:
        return

    # Compute base offsets
    # k_ptr/v_ptr layout: [total_kv, 8, 128]
    # k_exp_ptr/v_exp_ptr layout: [total_kv, 32, 128] but we only write for rows in [kv_start, kv_end)
    # We map original row r to expanded row idx = kv_start + r
    exp_row = kv_start + r

    # Loop over original kv heads (NUM_GROUPS = 8) and repeat to 32 heads
    for orig_h in range(NUM_GROUPS):
        # Load k_orig[r, orig_h, :] as 128-vector
        base = (r * (NUM_GROUPS * 128)) + orig_h * 128
        k_vec = tl.load(k_ptr + base)  # [128] float32

        # Write to k_exp[exp_row, h, :] for h = orig_h * REPEAT + rep
        for rep in range(REPEAT):
            h = orig_h * REPEAT + rep  # 0..31
            out_base = (exp_row * (32 * 128)) + h * 128
            tl.store(k_exp_ptr + out_base, k_vec)

    # Same for v
    for orig_h in range(NUM_GROUPS):
        base = (r * (NUM_GROUPS * 128)) + orig_h * 128
        v_vec = tl.load(v_ptr + base)
        for rep in range(REPEAT):
            h = orig_h * REPEAT + rep
            out_base = (exp_row * (32 * 128)) + h * 128
            tl.store(v_exp_ptr + out_base, v_vec)


@triton.jit
def per_query_attention_gqa_kernel(
    q_ptr,          # *float32, [total_q, 32, 128]
    k_exp_ptr,      # *float32, [total_kv, 32, 128]
    v_exp_ptr,      # *float32, [total_kv, 32, 128]
    out_ptr,        # *float32, [total_q, 32, 128]
    lse_ptr,        # *float32, [total_q, 32]
    qo_indptr_ptr,  # *int32, [len_indptr]
    kv_indptr_ptr,  # *int32, [len_indptr]
    sm_scale,       # float32
    NUM_SEGMENTS: tl.constexpr,  # number of segments
):
    """
    Triton kernel: one program per (i, h). It loops over segments b, computes 8 logits per
    segment, applies forward-causal mask, computes base-2 lse, softmax, and accumulates the
    output over d=0..127. This ensures Triton-only execution without host torch ops.
    """
    # Flatten pid: 0 .. (total_q * 32 - 1)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # If i >= Nq for some segment, we skip. We guard by checking q_end; but since grid is total_q,
    # for each b, i < Nq, so safe. We still guard against b loop Nq==0 segments.
    for b in range(NUM_SEGMENTS):
        qo_start = tl.load(qo_indptr_ptr + b)
        qo_end = tl.load(qo_indptr_ptr + b + 1)
        kv_start = tl.load(kv_indptr_ptr + b)
        kv_end = tl.load(kv_indptr_ptr + b + 1)

        if qo_end - qo_start == 0 or kv_end - kv_start == 0:
            continue

        # Load q[i, h, :] as 128-vector
        q_base = (qo_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_base)  # [128] float32

        # Compute 8 logits for this (i, h)
        logits = tl.zeros((8,), dtype=tl.float32)
        # Apply causal delta per segment
        delta = (kv_end - kv_start) - (qo_end - qo_start)

        # For each j (original kv head group), compute dot and apply mask
        for j in tl.static_range(0, 8):
            # Load k_exp[b, :, j, :] as 128-vector: rows r in [kv_start, kv_end)
            k_vec_j = tl.zeros((128,), dtype=tl.float32)
            for r in tl.static_range(0, 128):
                row_idx = kv_start + r
                if row_idx >= kv_end:
                    break
                base = (row_idx * (32 * 128)) + j * 128 + r * 128
                k_vec_j = k_vec_j + tl.load(k_exp_ptr + base)
            dot = tl.sum(q_vec * k_vec_j) * sm_scale
            # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
            if j >= (i + 1 + delta):
                dot = -float("inf")
            logits[j] = dot

        # Compute base-2 logsumexp over 8 positions
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions
        soft = tl.exp(logits - lse_val)  # [8] float32

        # Accumulate output[i, h, d] over d in 0..127
        out_base = (i * 32 + h) * 128
        for d in tl.static_range(0, 128):
            acc = tl.zeros((), dtype=tl.float32)
            for j in tl.static_range(0, 8):
                # v_exp[b, :, j, d]: rows r in [kv_start, kv_end)
                v_acc = tl.zeros((), dtype=tl.float32)
                for r in tl.static_range(0, 128):
                    row_idx = kv_start + r
                    if row_idx >= kv_end:
                        break
                    base = (row_idx * (32 * 128)) + j * 128 + r * 128 + d
                    v_acc += tl.load(v_exp_ptr + base)
                v_acc = v_acc * soft[j]
                acc = acc + v_acc
            tl.store(out_ptr + out_base + d, acc)

        # Store lse[i, h]
        lse_base = (i * 32 + h)
        tl.store(lse_ptr + lse_base, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward. All computation happens inside Triton kernels:
        - expand k and v to 32 heads (repeat_interleave in Triton).
        - compute per (i, h) attention across segments using Triton.
        Returns:
          - output: [total_q, 32, 128], bfloat16
          - lse: [total_q, 32], float32 (base-2 logsumexp)
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA for Triton"
        device = q.device

        # Ensure contiguous and cast to float32 for compute
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k.contiguous().to(torch.float32)
        v_f32 = v.contiguous().to(torch.float32)

        # Allocate expanded tensors
        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]
        # k_expanded and v_expanded: [total_kv, 32, 128]
        k_exp = torch.empty((total_kv, 32, 128), dtype=torch.float32, device=device)
        v_exp = torch.empty((total_kv, 32, 128), dtype=torch.float32, device=device)

        # Launch expand_k_v kernel: grid over (segments, rows in segment)
        NUM_GROUPS = 8
        REPEAT = 4
        grid_expand = (qo_indptr.numel(), total_kv)
        expand_k_v_kernel[grid_expand](
            k_f32, v_f32, k_exp, v_exp,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_GROUPS=NUM_GROUPS, REPEAT=REPEAT,
            num_warps=4, num_stages=2,
        )

        # Prepare output and lse
        out_f32 = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        NUM_SEGMENTS = qo_indptr.numel() - 1
        grid_attn = (total_q * 32,)
        per_query_attention_gqa_kernel[grid_attn](
            q_f32, k_exp, v_exp, out_f32, lse_f32,
            qo_indptr, kv_indptr, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
            num_warps=4, num_stages=2,
        )

        # Cast output to bfloat16 to match original
        out_bf16 = out_f32.to(torch.bfloat16)
        return out_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
