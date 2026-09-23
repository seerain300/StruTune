import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_gqa_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, q_end, kv_start, kv_end, sm_scale,
):
    """
    Triton kernel computing attention for one segment:
      - q: [q_end - q_start, 32, 128] float32
      - k: [kv_end - kv_start, 8, 128] float32
      - v: [kv_end - kv_start, 8, 128] float32
    Outputs:
      - out: [(q_end - q_start), 32, 128] float32
      - lse: [(q_end - q_start), 32] float32 (base-2 logsumexp)
    """
    Nq = q_end - q_start
    Nk = kv_end - kv_start
    delta = Nk - Nq

    # We process each (i, h) pair. Triton supports loops with tl.static_range but the ranges here are dynamic.
    # To be safe across Triton constraints, we use Python for-loops at call site and rely on Triton for elementwise work.
    # However, Triton requires static bounds for loops; since Nq is provided as runtime args, we cannot use tl.static_range directly here.
    # Hence, we structure the kernel to process one i and one h at a time via pointer arithmetic, which is valid.
    # For compatibility, we implement a simple i loop and h loop using Python loops in the host code. Inside Triton, we keep loops over j.

    # The kernel will be invoked once per segment. We place the i and h loops in the host. Here, we keep only j loop in Triton.
    pass


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-based implementation of the original run function.
        Returns output [total_q, 32, 128] bfloat16 and lse [total_q, 32] float32 (base-2 logsumexp).
        """
        device = q.device
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        # Sanity checks as in original
        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8 and q.shape[2] == 128
        assert total_q == qo_indptr[-1].item()
        assert total_kv == kv_indptr[-1].item()

        out = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)  # accumulate in float32
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        B = qo_indptr.shape[0] - 1
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice and cast to float32, make contiguous
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)  # [Nq, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32)  # [Nk, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32)  # [Nk, 8, 128]

            # We need q as [Nq*32, 128] for linear addressing in Triton
            q_flat = q_batch.view(-1, 128).contiguous()  # [Nq*32, 128]
            k_flat = k_batch.view(-1, 128).contiguous()  # [Nk*8, 128], but we only need [Nk, 8, 128] linearly
            v_flat = v_batch.view(-1, 128).contiguous()  # [Nk*8, 128]

            # Process each (i, h) pair in Python loops; Triton handles j loop
            Nq = q_end - q_start
            for i in range(Nq):
                for h in range(32):
                    # Initialize logits vector for this (i, h): shape [8]
                    logits = tl.full((8,), -float("inf"), tl.float32)
                    # Compute dot products for j in [0..7] with proper causal mask
                    for j in range(8):
                        orig_h = h % 8
                        # Load q vector: q[i, h, :]
                        q_lin = i * (32 * 128) + h * 128  # since q_flat is [Nq*32, 128]
                        q_vec = tl.load(q_ptr + q_lin)  # Triton pointer arithmetic expects q_ptr as input
                        # Load k vector: k[kv_start + j, orig_h, :]
                        k_lin = (kv_start + j) * (8 * 128) + orig_h * 128  # k_flat is [Nk*8, 128]
                        k_vec = tl.load(k_ptr + k_lin)
                        # Dot product
                        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
                        # Causal mask: allow kv position j if j < (i + 1 + delta)
                        if not (j < (i + 1 + delta)):
                            dot = -float("inf")
                        # Store into logits vector for this h
                        # Triton doesn't support direct tl.store to a register vector with index; we manually place.
                        # We keep logits as a list-like (actually Triton vector) and overwrite the j-th element by recomputing via masking.
                        # Simpler: reconstruct vector with tl.where
                        j_vec = tl.arange(0, 8)
                        logits = tl.where(j_vec == j, dot, logits)

                    # Compute base-2 logsumexp over 8 positions
                    m = tl.max(logits, axis=0)
                    sum_exp = tl.sum(tl.exp(logits - m), axis=0)
                    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

                    # Softmax across 8 positions
                    soft = tl.exp(logits - lse_val)  # [8], float32

                    # Accumulate output[i, h, :] += soft[j] * v[kv_start + j, orig_h, :] for j in 0..7
                    for j in range(8):
                        orig_h = h % 8
                        v_lin = (kv_start + j) * (8 * 128) + orig_h * 128
                        v_vec = tl.load(v_ptr + v_lin)  # [128]
                        out_lin = (i * 32 + h) * 128
                        # output is float32; store adds to existing
                        # We must ensure atomic add if multiple programs; but grid=1 here. We can store and atomically add if needed.
                        # Simpler: write once since we iterate with unique i,h.
                        tl.store(out_ptr + out_lin, v_vec * soft[j])

                    # Store lse[i, h]
                    lse_lin = (i * 32 + h)
                    tl.store(lse_ptr + lse_lin, lse_val)

        # Cast output to bfloat16 to match original, keep lse float32
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
