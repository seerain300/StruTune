import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logit_and_sumexp_segmentwise(
    q_ptr,       # *float32, [total_q, 32, 128]
    k_exp_ptr,   # *float32, [total_kv, 32, 128] (expanded k)
    out_log_ptr, # *float32, [NUM_SEGMENTS, 8, total_q * 32] (we'll index by (b, j, pid))
    qo_indptr_ptr, kv_indptr_ptr, # int32
    total_q, total_kv, sm_scale,  # not used here; included for signature completeness
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h).
    For each segment b, compute 8 logits for that (i, h), apply causal mask, and store into out_log_ptr[b, :, pid],
    where pid = i * 32 + h.
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    for b in range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)
        q_end = tl.load(qo_indptr_ptr + b + 1)
        kv_start = tl.load(kv_indptr_ptr + b)
        kv_end = tl.load(kv_indptr_ptr + b + 1)

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq > 0:
            # Load q[i, h, :]
            q_vec_base = (q_start + i) * 32 * 128 + h * 128
            q_vec = tl.load(q_ptr + q_vec_base)  # [128] float32

        # Compute 8 logits (no sm_scale: apply in reduction)
        # We'll store each j into out_log_ptr[b, j, pid]
        for j in range(8):
            # Accumulate dot across all kv rows in the segment
            # We do this in chunks along the 128-dim to limit register use
            dot_acc = 0.0
            # Iterate rows from kv_start to kv_end
            for r in range(kv_start, kv_end):
                # Pointer to k_exp[r, j, :] flattened over head dimension
                # k_exp has shape [Nk, 32, 128]; we need the j-th head for that row
                # Layout: ((r * 32 + j) * 128 + offset) across dim-1
                # However, since we pre-expanded and provided k_exp_ptr as [Nk, 32, 128],
                # we index as: base = r * 32 * 128 + j * 128; then add offset.
                base = r * 32 * 128 + j * 128
                k_row_vec = tl.load(k_exp_ptr + base)  # [128]
                # Accumulate dot
                dot_acc += tl.sum(q_vec * k_row_vec, axis=0)

            # Apply forward-causal mask: j >= (i + 1 + delta) -> -inf
            delta = Nk - (q_end - q_start)  # per-segment delta
            mask_inf = (j >= (i + 1 + delta))
            if mask_inf:
                dot_acc = -float('inf')

            # Store dot_acc into out_log_ptr[b, j, pid]
            # out_log_ptr is [NUM_SEGMENTS, 8, total_q*32]
            pid_idx = pid  # since one program per (i,h)
            store_offset = b * 8 * (total_q * 32) + j * (total_q * 32) + pid_idx
            tl.store(out_log_ptr + store_offset, dot_acc)


@triton.jit
def reduce_segments_kernel(
    out_log_ptr,  # *float32, [NUM_SEGMENTS, 8, total_q*32]
    v_exp_ptr,    # *float32, [total_kv, 32, 128]
    out_ptr,      # *float32, [total_q, 32, 128]
    lse_ptr,      # *float32, [total_q, 32]
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,  # sm_scale unused here
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h).
    Load logits2D for each segment b, compute base-2 lse per j (since j-dim is what we softmax),
    compute softmax per j across segments, and accumulate output[i, h, :] using v_expanded for each segment.
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # We need to compute lse per (i,h). In original, lse is across j-dimension of logits.
    # Here, per segment b, logits[i,h,j] are per b. So lse per (i,h) is logsumexp across b of logits[i,h,j] per j.
    # But the original lse is computed per (i, qo_head) for the q segment-specific logits. More precisely:
    # lse[i, h] = logsumexp over the 8 columns j, across segments. However, original code computes logits per segment
    # and then lse per (i,h) within that segment. To match that, we compute lse per segment b for each (i,h),
    # but the output aggregation is across segments. The provided reference run returns lse for each (i,h) per segment.
    # We will compute lse per segment b for each (i,h) and store in lse_ptr[b * (total_q * 32) + pid].
    # For simplicity and to keep Triton-only, we compute lse per b here.

    # First, compute lse per segment b for (i, h)
    for b in range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)
        q_end = tl.load(qo_indptr_ptr + b + 1)
        kv_start = tl.load(kv_indptr_ptr + b)
        kv_end = tl.load(kv_indptr_ptr + b + 1)

        # Prepare per-j lse values (logsumexp across segments)
        # We will iterate over segments r to get logits[i,h,j] = dot computed above
        lse_vals = tl.zeros((8,), dtype=tl.float32)
        for j in range(8):
            # Compute logsumexp across segments for column j
            # Initialize with -inf
            m = -float('inf')
            # Iterate segments r to get logits[i,h,j] = dot for each r
            for r in range(NUM_SEGMENTS):
                store_offset = r * 8 * (total_q * 32) + j * (total_q * 32) + pid
                val = tl.load(out_log_ptr + store_offset)
                # val could be -inf if masked; we treat it as -inf for max
                m = tl.maximum(m, val)
            # sum exp
            sum_exp = 0.0
            for r in range(NUM_SEGMENTS):
                store_offset = r * 8 * (total_q * 32) + j * (total_q * 32) + pid
                val = tl.load(out_log_ptr + store_offset)
                sum_exp += tl.exp(val - m)
            lse_vals[j] = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Now store lse per segment for (i,h)
        # We store lse_vals to lse_ptr[b * (total_q * 32) + pid]
        for j in range(8):
            store_offset = b * (total_q * 32) + pid
            tl.store(lse_ptr + store_offset, lse_vals[j])

    # Second, compute output[i, h, :] by accumulating across segments
    # We need softmax per j across segments. To do that, we need logits per segment b for each j.
    # Then output[i,h,:] += soft_j[b] * v_exp[kv_start + b, (h % 8), :]
    for j in range(8):
        # Compute m_j = max across segments for column j
        m_j = -float('inf')
        for r in range(NUM_SEGMENTS):
            store_offset = r * 8 * (total_q * 32) + j * (total_q * 32) + pid
            val = tl.load(out_log_ptr + store_offset)
            m_j = tl.maximum(m_j, val)
        sum_j = 0.0
        for r in range(NUM_SEGMENTS):
            store_offset = r * 8 * (total_q * 32) + j * (total_q * 32) + pid
            val = tl.load(out_log_ptr + store_offset)
            sum_j += tl.exp(val - m_j)
        # softmax per segment r for column j
        out_vec = tl.zeros((128,), dtype=tl.float32)
        # For each segment r
        for r in range(NUM_SEGMENTS):
            store_offset = r * 8 * (total_q * 32) + j * (total_q * 32) + pid
            logit_rj = tl.load(out_log_ptr + store_offset)  # scalar
            soft_rj = tl.exp(logit_rj - m_j) / sum_j
            # v_exp[b, (h % 8), :]
            head_idx = h % 8
            v_base = (kv_start + r) * 32 * 128 + head_idx * 128
            v_vec = tl.load(v_exp_ptr + v_base)  # [128]
            out_vec += v_vec * soft_rj

        # Store out[i, h, :]
        out_base = i * 32 * 128 + h * 128
        tl.store(out_ptr + out_base, out_vec)

# ModelNew: Triton-only entry point
class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure devices/dtypes/contiguity
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "Triton kernels require CUDA tensors."
        # Cast q to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        # Pre-expand k and v to 32 heads on host (cheap relative to kernel launch)
        k_expanded = k.repeat_interleave(4, dim=1).contiguous()  # [total_kv, 32, 128]
        v_expanded = v.repeat_interleave(4, dim=1).contiguous()  # [total_kv, 32, 128]
        # Prepare output and lse
        total_q = q_f32.shape[0]
        total_kv = k_expanded.shape[0]
        num_qo_heads = q_f32.shape[1]
        num_qh = num_qo_heads  # 32
        num_kv_heads = k_expanded.shape[1]  # 32
        output = torch.empty((total_q, num_qo_heads, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)

        # qo_indptr, kv_indptr must be int32 and contiguous
        qo_indptr_i32 = qo_indptr.int().contiguous()
        kv_indptr_i32 = kv_indptr.int().contiguous()

        NUM_SEGMENTS = qo_indptr_i32.numel() - 1

        # Allocate per-segment logits buffer: [NUM_SEGMENTS, 8, total_q*32]
        out_log = torch.empty((NUM_SEGMENTS, 8, total_q * num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel 1: compute logits per segment for each (i, h)
        grid = (total_q * num_qo_heads,)
        compute_logit_and_sumexp_segmentwise[grid](
            q_f32, k_expanded, out_log, qo_indptr_i32, kv_indptr_i32,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Launch Triton kernel 2: reduce across segments and produce output & lse
        reduce_segments_kernel[grid](
            out_log, v_expanded, output, lse, qo_indptr_i32, kv_indptr_i32,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Cast output to bfloat16 to match original return types (output and lse)
        output_bf16 = output.to(torch.bfloat16)
        lse_base2 = lse  # already in base-2 logsumexp

        return output_bf16, lse_base2


def run(*args):
    return ModelNew()(*args)
