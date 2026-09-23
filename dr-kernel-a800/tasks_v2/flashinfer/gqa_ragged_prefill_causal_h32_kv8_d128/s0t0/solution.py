import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, kv_start, q_end, kv_end, sm_scale,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Batch segment boundaries
    Nq = q_end - q_start
    Nk = kv_end - kv_start

    # We will process each query token index q_index in [0, Nq)
    # and compute attention for 32 heads (BLOCK_H=32, head_dim=128, BLOCK_D=128).
    # For each head h in 0..31, we compute logits across 32 expanded kv heads (j in 0..31),
    # apply mask, compute lse, softmax, and accumulate output.

    # Precompute constants
    inv_sqrt_d = sm_scale  # 1 / sqrt(128)
    log2 = 1.4426950408889634  # 1 / ln(2)

    # For each query position
    for q_index in range(0, Nq):
        # Base offsets
        # q layout: [Nq, 32, 128], contiguous
        # We'll iterate heads h in 0..31
        for h in range(0, 32):
            # Row base for q at (q_index, h)
            q_row_base = q_ptr + (q_start + q_index) * 32 * 128 + h * 128
            # Prepare logits vector for head h, across j=0..31 (expanded heads)
            logits_vec = tl.full((32,), -float('inf'), dtype=tl.float32)

            # Compute logits[h, j] for j in 0..31
            for j in range(0, 32):
                orig_h = j % 8  # map expanded head to original 8 kv heads
                # k_row for orig_h
                k_row_base = k_ptr + (kv_start + j) * 128 + orig_h * 128
                # v_row for orig_h
                v_row_base = v_ptr + (kv_start + j) * 128 + orig_h * 128

                # dot product across 128 dims
                acc = 0.0
                # Load q vector for this head
                q_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
                for d in range(0, BLOCK_D):
                    q_off = q_row_base + d
                    q_val = tl.load(q_off, mask=True, other=0.0)
                    q_val = q_val.to(tl.float32)
                    q_vec[d] = q_val

                # Load k vector for this orig head at token j
                k_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
                for d in range(0, BLOCK_D):
                    k_off = k_row_base + d
                    k_val = tl.load(k_off, mask=True, other=0.0)
                    k_val = k_val.to(tl.float32)
                    k_vec[d] = k_val

                # Accumulate dot product
                for d in range(0, BLOCK_D):
                    acc += q_vec[d] * k_vec[d]

                # Scale
                acc = acc * inv_sqrt_d
                # Causal mask: allow j if j < (q_index + 1 + (Nk - Nq))
                mask_j = (j < (q_index + 1 + (Nk - Nq)))
                # If masked, set to -inf
                acc = tl.where(mask_j, acc, -float('inf'))
                # Store into logits vector at position j
                logits_vec[j] = acc

            # Compute lse for this (q_index, h) in base-2
            max_logits = tl.max(logits_vec, axis=0)
            sum_exp = 0.0
            for j in range(0, 32):
                sum_exp += tl.exp(logits_vec[j] - max_logits)
            lse_base = tl.log(sum_exp) + max_logits  # logsumexp in natural log
            lse_base2 = lse_base / log2
            # Store lse to lse_ptr[q_index, h]
            lse_off = lse_ptr + q_start * 32 + q_index * 32 + h
            tl.store(lse_off, lse_base2)

            # Compute softmax across j for this head h
            sum_softmax = 0.0
            for j in range(0, 32):
                e = tl.exp(logits_vec[j] - lse_base2)
                sum_softmax += e
            out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
            # Accumulate output: out[h] += softmax[j] * v_row(orig_h, token j) for j in 0..31
            for j in range(0, 32):
                e = tl.exp(logits_vec[j] - lse_base2)
                prob = e / sum_softmax
                # Load v row for orig head orig_h at token j
                v_row_base = v_ptr + (kv_start + j) * 128 + (j % 8) * 128
                v_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
                for d in range(0, BLOCK_D):
                    v_off = v_row_base + d
                    v_val = tl.load(v_off, mask=True, other=0.0).to(tl.float32)
                    v_vec[d] = v_val
                out_vec += prob * v_vec

            # Store output row for head h at query position q_index
            out_base = out_ptr + (q_start + q_index) * 32 * 128 + h * 128
            for d in range(0, BLOCK_D):
                tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16 (we'll cast to fp32 for compute)
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32 cumulative
        kv_indptr: [len_indptr], int32 cumulative
        sm_scale: float32 scalar (e.g., 1/sqrt(128))
        Returns: (output: [total_q, 32, 128], lse: [total_q, 32])
        """
        # Ensure on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA device for Triton."
        # Cast to fp32 for computation
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k.contiguous().to(torch.float32)
        v_f32 = v.contiguous().to(torch.float32)

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]
        assert q_f32.shape[1] == 32 and q_f32.shape[2] == 128, "q must have [N, 32, 128]"
        assert k_f32.shape[1] == 8 and k_f32.shape[2] == 128, "k must have [M, 8, 128]"
        assert v_f32.shape == k_f32.shape, "v must match k shape [M, 8, 128]"
        assert qo_indptr.shape[0] > 0 and kv_indptr.shape[0] > 0
        assert qo_indptr[-1].item() == total_q and kv_indptr[-1].item() == total_kv

        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Launch Triton kernel per batch segment
        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Launch kernel for this batch segment
            # Grid: one program handles one query index; but Triton kernels are best with 2D tiling.
            # Here Nq is dynamic; we can still launch with grid (1,), and loop in kernel.
            # However, Triton prefers specifying grid and BLOCK sizes. We'll choose grid=(1,) and loop Nq inside.
            attention_forward[(1,)](
                q_f32, k_f32, v_f32,
                output, lse,
                q_start, kv_start, q_end, kv_end, sm_scale,
                BLOCK_H=32, BLOCK_D=128
            )

        # Cast output back to bfloat16 as original returns
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
