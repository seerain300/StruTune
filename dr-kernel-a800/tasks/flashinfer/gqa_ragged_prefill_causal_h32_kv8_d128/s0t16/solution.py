import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_segment(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, kv_start, q_end, kv_end, sm_scale,
    BLOCK_D: tl.constexpr
):
    """
    Triton kernel computing attention for one segment defined by [q_start:q_end) queries and [kv_start:kv_end) keys/values.
    Inputs are already sliced and contiguous on device. All math is in float32.
    """
    Nq = q_end - q_start
    Nk = kv_end - kv_start

    # Loop over query tokens
    for q_index in tl.static_range(0, Nq):
        # Loop over 32 query heads
        for h in tl.static_range(0, 32):
            # Base pointer for this (q_index, h)
            q_row_base = q_ptr + (q_start + q_index) * 32 * BLOCK_D + h * BLOCK_D

            # Vector to hold logits across 32 expanded kv heads (only j in 0..7 contribute, others -inf)
            logits_vec = tl.full((32,), -float('inf'), dtype=tl.float32)

            # Compute dot products for original kv heads j=0..7 and map to query head h (GQA: orig_h = h % 8)
            for j in tl.static_range(0, 8):
                orig_h = h % 8
                kv_position = kv_start + j

                # Load q vector for head h (128 elements)
                q_off_d = q_row_base + tl.arange(0, BLOCK_D)
                q_vec = tl.load(q_off_d, mask=tl.arange(0, BLOCK_D) < BLOCK_D, other=0.0).to(tl.float32)

                # Load k vector for kv_position and head orig_h (128 elements)
                k_row_base = k_ptr + kv_position * 8 * BLOCK_D + orig_h * BLOCK_D
                k_off_d = k_row_base + tl.arange(0, BLOCK_D)
                k_vec = tl.load(k_off_d, mask=tl.arange(0, BLOCK_D) < BLOCK_D, other=0.0).to(tl.float32)

                # Dot product over 128 dims
                acc = tl.sum(q_vec * k_vec, axis=0)

                # Scale by sm_scale
                acc = acc * sm_scale

                # Apply forward-causal mask: allow j if j < (q_index + 1 + (Nk - Nq))
                j_vec = tl.arange(0, 32)
                mask_j = j_vec < (q_index + 1 + (Nk - Nq))
                acc = tl.where(mask_j, acc, -float('inf'))

                # Place acc into logits_vec at position j
                logits_vec[j] = acc

            # Compute base-2 logsumexp over expanded heads
            max_logits = tl.max(logits_vec, axis=0)
            sum_exp = 0.0
            for jj in tl.static_range(0, 32):
                sum_exp += tl.exp(logits_vec[jj] - max_logits)
            lse_base = tl.log(sum_exp) + max_logits  # natural logsumexp
            lse_base2 = lse_base / 1.4426950408889634  # 1/ln(2)

            # Store lse to lse_ptr[q_index, h]
            lse_off = lse_ptr + q_start * 32 + q_index * 32 + h
            tl.store(lse_off, lse_base2)

            # Softmax across expanded heads (j=0..31)
            sum_softmax = 0.0
            for jj in tl.static_range(0, 32):
                e = tl.exp(logits_vec[jj] - lse_base2)
                sum_softmax += e

            # Output vector for this head
            out_row_base = out_ptr + (q_start + q_index) * 32 * BLOCK_D + h * BLOCK_D
            out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)

            # Accumulate output using v for allowed j
            for j in tl.static_range(0, 8):
                kv_position = kv_start + j
                orig_h = h % 8
                v_row_base = v_ptr + kv_position * 8 * BLOCK_D + orig_h * BLOCK_D
                v_off_d = v_row_base + tl.arange(0, BLOCK_D)
                v_vec = tl.load(v_off_d, mask=tl.arange(0, BLOCK_D) < BLOCK_D, other=0.0).to(tl.float32)

                e = tl.exp(logits_vec[j] - lse_base2)  # softmax for this j
                out_vec += e * v_vec

            # Store out[q_index, h, :]
            out_off_d = out_row_base + tl.arange(0, BLOCK_D)
            tl.store(out_off_d, out_vec, mask=tl.arange(0, BLOCK_D) < BLOCK_D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-based forward that matches the simplified spec.
        Returns:
          output: [total_q, 32, 128], bfloat16
          lse: [total_q, 32], float32
        """
        # Sanity checks
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        total_q, Q_heads, D = q.shape
        total_kv, K_heads, Dk = k.shape
        assert Q_heads == 32 and D == 128 and K_heads == 8 and Dk == 128
        assert qo_indptr[-1].item() == total_q
        assert kv_indptr[-1].item() == total_kv

        # Prepare outputs (float32 for computation)
        out = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Iterate over segments b
        B = qo_indptr.numel() - 1
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice and ensure contiguity
            q_batch = q[q_start:q_end].contiguous()
            k_batch = k[kv_start:kv_end].contiguous()
            v_batch = v[kv_start:kv_end].contiguous()

            # Cast to float32 for kernel math
            q_batch = q_batch.to(torch.float32)
            k_batch = k_batch.to(torch.float32)
            v_batch = v_batch.to(torch.float32)

            # Launch Triton kernel for this segment
            attention_forward_segment[(1,)](
                q_batch, k_batch, v_batch, out, lse,
                q_start, kv_start, q_end, kv_end, sm_scale,
                BLOCK_D=128,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 as in the original spec
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
