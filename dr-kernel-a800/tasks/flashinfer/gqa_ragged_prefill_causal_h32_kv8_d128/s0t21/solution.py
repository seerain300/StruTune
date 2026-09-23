import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_gqa(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, q_end, kv_start, kv_end, sm_scale,
):
    """
    Triton kernel that computes attention for one segment:
      q_batch: [q_end - q_start, 32, 128] (passed as base+strides)
      k_batch: [kv_end - kv_start, 8, 128] (passed as base+strides)
      v_batch: [kv_end - kv_start, 8, 128] (passed as base+strides)
      Outputs:
        out: [(q_end - q_start), 32, 128] float32
        lse: [(q_end - q_start), 32] float32 (base-2 logsumexp)
    """
    Nq = q_end - q_start
    Nk = kv_end - kv_start
    delta = Nk - Nq

    # We process each (i, h) pair; Triton uses static_range. We guard with if to avoid out-of-range.
    for i in tl.static_range(0, 1024):
        if i >= Nq:
            break
        for h in tl.static_range(0, 32):
            # Base pointer for q[i, h, :]
            q_row_base = q_ptr + (q_start + i) * 32 * 128 + h * 128
            q_vec = tl.load(q_row_base + tl.arange(0, 128), mask=tl.arange(0, 128) < 128, other=0.0).to(tl.float32)  # [128]

            # Prepare 32x8 logits matrix for this (i, h)
            logits_mat = tl.full((32, 8), -float('inf'), dtype=tl.float32)

            # Compute dot products for j in 0..7; map to orig_h = h % 8
            for j in tl.static_range(0, 8):
                orig_h = h % 8
                k_row_base = k_ptr + (kv_start + j) * 8 * 128 + orig_h * 128
                k_vec = tl.load(k_row_base + tl.arange(0, 128), mask=tl.arange(0, 128) < 128, other=0.0).to(tl.float32)  # [128]
                v_row_base = v_ptr + (kv_start + j) * 8 * 128 + orig_h * 128
                v_vec = tl.load(v_row_base + tl.arange(0, 128), mask=tl.arange(0, 128) < 128, other=0.0).to(tl.float32)  # [128]

                dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar

                # Forward-causal mask: allow j if j < (i + 1 + delta)
                mask_j = j < (i + 1 + delta)
                logits_mat[h, j] = tl.where(mask_j, dot, -float('inf'))

            # Compute base-2 logsumexp over the 8 columns
            logits_vec = logits_mat[:, :8].to(tl.float32).reshape((8,))
            max_logit = tl.max(logits_vec, axis=0)
            sum_exp = tl.sum(tl.exp(logits_vec - max_logit), axis=0)
            lse_base = tl.log(sum_exp) + max_logit  # natural logsumexp
            lse_base2 = lse_base / 1.4426950408889634  # 1/ln(2)

            # Store lse[i, h]
            lse_off = lse_ptr + (q_start + i) * 32 + h
            tl.store(lse_off, lse_base2)

            # Softmax over the 8 positions
            exps = tl.exp(logits_vec - lse_base2)  # [8]
            sum_softmax = tl.sum(exps, axis=0)

            # Accumulate output[i, h, :] = sum_j softmax[j] * v[kv_start + j, orig_h, :]
            out_row_base = out_ptr + (q_start + i) * 32 * 128 + h * 128
            out_vec = tl.zeros((128,), dtype=tl.float32)
            for j in tl.static_range(0, 8):
                orig_h = h % 8
                v_row_base_j = v_ptr + (kv_start + j) * 8 * 128 + orig_h * 128
                v_vec_j = tl.load(v_row_base_j + tl.arange(0, 128), mask=tl.arange(0, 128) < 128, other=0.0).to(tl.float32)  # [128]
                out_vec += (exps[j] / sum_softmax) * v_vec_j

            # Store out[i, h, :]
            tl.store(out_row_base + tl.arange(0, 128), out_vec, mask=tl.arange(0, 128) < 128)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-based forward that matches the original PyTorch behavior:
          - Slices q/k/v per segment.
          - Computes attention with GQA mapping: each qo head h maps to kv head orig_h = h % 8.
          - Applies forward-causal mask j < (i + 1 + (Nk - Nq)).
          - Computes base-2 logsumexp and softmax across 8 kv positions per (i, h).
          - Accumulates output [N, 32, 128] bfloat16 and lse [N, 32] float32.
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
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
            segment_attention_gqa[(1,)](
                q_batch, k_batch, v_batch, out, lse,
                q_start, q_end, kv_start, kv_end, sm_scale,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 as in the original spec
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
