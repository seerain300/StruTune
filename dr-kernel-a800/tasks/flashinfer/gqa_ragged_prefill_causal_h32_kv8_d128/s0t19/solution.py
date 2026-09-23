import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_segment_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    sm_scale: tl.float32,
    # Host will pass total_q, total_kv, and segment bounds via qo_indptr/kv_indptr loads
):
    # One program per segment
    b = tl.program_id(0)

    # Load segment bounds (indices are int32)
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    # Number of tokens in this segment (runtime integers)
    Nq = q_end - q_start
    Nk = kv_end - kv_start

    # Constants
    Q_HEADS = 32
    K_HEADS = 8
    D = 128
    LN2 = 0.6931471805599453  # ln(2)

    # Precompute allowed kv positions per query token: allowed_j = min(Nk-1, i + 1 + (Nk - Nq))
    # We'll use this in mask application.

    # We'll implement loops explicitly with runtime bounds (avoid tl.static_range to be safe).
    # For each query token i
    for i in range(0, Nq):
        allowed_j = i + 1 + (Nk - Nq)
        if allowed_j < 0:
            allowed_j = 0
        if allowed_j > Nk - 1:
            allowed_j = Nk - 1

        # Build logits for this query token i across all query heads h and kv positions j
        # logits[h, j] = dot(q[i, h, :], k[kv_start + j, (h % 8), :]) * sm_scale
        # We'll store logits in a 32x32 matrix using pointer arithmetic and scalar loads.

        # First, compute q_vec for all h and store q elements to avoid recomputing
        # We will load q components directly in softmax computation; we don't materialize q_vec as a tensor.
        # Compute logits[h, j] directly:
        # Initialize per-head max for LSE
        row_max = tl.full((Q_HEADS,), -float('inf'), dtype=tl.float32)

        # For each query head h
        for h in range(0, Q_HEADS):
            # Initialize logit row accumulator
            for j in range(0, Nk):
                # Initialize scalar accumulator for this (h, j)
                logits_hj = 0.0
                # Dot product over D
                for d in range(0, D):
                    q_off = (((q_start + i) * Q_HEADS + h) * D) + d
                    q_elem = tl.load(q_ptr + q_off).to(tl.float32)
                    k_orig_h = (h % K_HEADS)  # 0..7
                    k_off = (((kv_start + j) * K_HEADS + k_orig_h) * D) + d
                    k_elem = tl.load(k_ptr + k_off).to(tl.float32)
                    logits_hj += q_elem * k_elem
                # Apply scaling
                logits_hj *= sm_scale
                # Store logits[h, j] into a flat buffer. We need a flat index for (h, j).
                # We'll use a preallocated logits float32 array of size Q_HEADS*Nk on host and pass a pointer.
                # However, to avoid dynamic tensor creation in Triton, we will compute softmax and output directly without materializing logits.
                # Instead, we will recompute q_vec per softmax pass. To do softmax, we need max; compute max per head now.
                row_max[h] = tl.maximum(row_max[h], logits_hj)

        # Now compute softmax over j for each h and accumulate output. We'll recompute dot products as needed.
        # We'll use two passes: first to compute max, then to compute sum_exp and write output. Since Triton loops are runtime,
        # we can't easily vectorize, but we can compute sum_exp and output without materializing logits.
        # Pass 2: compute sum_exp and write output
        row_sum_exp = tl.zeros((Q_HEADS,), dtype=tl.float32)
        for h in range(0, Q_HEADS):
            sum_exp_h = 0.0
            for j in range(0, Nk):
                # Recompute logits_hj
                logits_hj = 0.0
                for d in range(0, D):
                    q_off = (((q_start + i) * Q_HEADS + h) * D) + d
                    q_elem = tl.load(q_ptr + q_off).to(tl.float32)
                    k_orig_h = (h % K_HEADS)
                    k_off = (((kv_start + j) * K_HEADS + k_orig_h) * D) + d
                    k_elem = tl.load(k_ptr + k_off).to(tl.float32)
                    logits_hj += q_elem * k_elem
                logits_hj *= sm_scale
                # Mask: if j >= allowed_j, set to -inf for softmax
                if j >= allowed_j:
                    logits_hj = -float('inf')
                sum_exp_h += tl.exp(logits_hj - row_max[h])
            row_sum_exp[h] = sum_exp_h

        inv_row_sum = 1.0 / row_sum_exp

        # Now write output for each query head h: out[i, h, :] = sum_j attn[h, j] * v[kv_start + j, (h % 8), :]
        for h in range(0, Q_HEADS):
            out_row = tl.zeros((D,), dtype=tl.float32)
            for j in range(0, Nk):
                logits_hj = 0.0
                for d in range(0, D):
                    q_off = (((q_start + i) * Q_HEADS + h) * D) + d
                    q_elem = tl.load(q_ptr + q_off).to(tl.float32)
                    k_orig_h = (h % K_HEADS)
                    k_off = (((kv_start + j) * K_HEADS + k_orig_h) * D) + d
                    k_elem = tl.load(k_ptr + k_off).to(tl.float32)
                    logits_hj += q_elem * k_elem
                logits_hj *= sm_scale
                # Mask
                if j >= allowed_j:
                    attn_hj = 0.0
                else:
                    attn_hj = tl.exp(logits_hj - row_max[h]) * inv_row_sum[h]
                # v vector for this kv position and original kv head
                k_orig_h_v = j % K_HEADS
                v_vec = tl.zeros((D,), dtype=tl.float32)
                for d in range(0, D):
                    v_off = (((kv_start + j) * K_HEADS + k_orig_h_v) * D) + d
                    v_elem = tl.load(v_ptr + v_off).to(tl.float32)
                    v_vec[d] = v_elem
                out_row += attn_hj * v_vec

            # Write out[i, h, :]
            out_base = ((q_start + i) * Q_HEADS + h) * D
            for d in range(0, D):
                out_ptr[out_base + d] = out_row[d]

        # Compute LSE in base-2: lse[i, h] = logsumexp(logits[i, h]) / ln(2)
        for h in range(0, Q_HEADS):
            sum_exp_lse = 0.0
            for j in range(0, Nk):
                logits_hj = 0.0
                for d in range(0, D):
                    q_off = (((q_start + i) * Q_HEADS + h) * D) + d
                    q_elem = tl.load(q_ptr + q_off).to(tl.float32)
                    k_orig_h = (h % K_HEADS)
                    k_off = (((kv_start + j) * K_HEADS + k_orig_h) * D) + d
                    k_elem = tl.load(k_ptr + k_off).to(tl.float32)
                    logits_hj += q_elem * k_elem
                logits_hj *= sm_scale
                if j >= allowed_j:
                    logits_hj = -float('inf')
                sum_exp_lse += tl.exp(logits_hj - row_max[h])
            lse_val = row_max[h] + tl.log(sum_exp_lse)
            lse_val = lse_val / LN2
            lse_ptr[(q_start + i) * Q_HEADS + h] = lse_val

    # No return; we write out_ptr and lse_ptr directly.


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [N, 32, 128], bfloat16
        k: [N, 8, 128], bfloat16
        v: [N, 8, 128], bfloat16
        qo_indptr: [B+1], int32
        kv_indptr: [B+1], int32
        sm_scale: float32 scalar
        Returns:
        output: [total_q, 32, 128], bfloat16
        lse: [total_q, 32], float32
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA for Triton."
        device = q.device

        # Ensure inputs are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        B = qo_indptr.shape[0] - 1  # number of segments

        # Output tensors (float32 for compute; cast later to bfloat16)
        out = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Launch one program per segment
        grid = (B,)
        gqa_segment_kernel[grid](
            q, k, v,
            out, lse,
            qo_indptr, kv_indptr,
            sm_scale,
            num_warps=4,
        )

        # Cast output to bfloat16 to match original
        out = out.to(torch.bfloat16)
        return out, lse


def run(*args):
    return ModelNew()(*args)
