import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_segment(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, kv_start, q_end, kv_end, sm_scale,
    # q: [Nq, 32, 128], k: [Nk, 8, 128], v: [Nk, 8, 128]
    # out: [Nq, 32, 128], lse: [Nq, 32]
    Nq: tl.constexpr, Nk: tl.constexpr,
):
    # Compile-time dimensions (host ensures consistency)
    D = 128
    Q_HEADS = 32
    K_HEADS = 8
    LN2 = 0.6931471805599453  # ln(2)

    # Loop over query tokens and query heads
    for i in tl.static_range(0, Nq):
        for h in tl.static_range(0, Q_HEADS):
            # Compute logits across 32 expanded kv heads (8 original kv heads, each repeated 4x)
            # Each expanded head j corresponds to original kv head orig_h = j % K_HEADS
            logits = tl.full((32,), -1.0e20, dtype=tl.float32)

            # Accumulate dot products for j in [0..7]
            for j in tl.static_range(0, K_HEADS):
                orig_h = j  # GQA mapping: query head h maps to kv head orig_h = h % K_HEADS
                # Load q[i, h, :] -> shape [D]
                q_vec = tl.load(q_ptr + i * (Q_HEADS * D) + h * D + tl.arange(0, D))
                # Load k[kv_start + j, orig_h, :] -> shape [D]
                k_vec = tl.load(k_ptr + (kv_start + j) * (K_HEADS * D) + orig_h * D + tl.arange(0, D))
                # Compute dot product
                dot = tl.sum(q_vec * k_vec, axis=0)
                # Scale
                logit_val = dot * sm_scale
                # Position index in logits vector
                pos = h * K_HEADS + j
                # Update logits
                logits[pos] = logit_val

            # Apply forward-causal mask: j allowed if j < (i + 1 + (Nk - Nq)), i.e., j < min(31, i + 1 + delta)
            # delta = Nk - Nq
            delta = Nk - Nq
            # Compute allowed max j for this query position i
            max_j = i + 1 + delta
            # Truncate to 32
            max_j = tl.minimum(max_j, 31)
            # Mask invalid positions
            # Triton supports elementwise mask on tl.arange via broadcasting
            j_idx = tl.arange(0, 32)
            mask_j = j_idx < max_j
            logits = tl.where(mask_j, logits, -1.0e20)

            # Compute logsumexp over 32 positions in natural log
            lse_val = tl.logsumexp(logits)
            # Convert to base-2
            lse_val = lse_val / LN2

            # Store lse[i, h]
            lse_ptr[i * Q_HEADS + h] = lse_val

            # Compute softmax over 32 positions and accumulate output[i, h, :]
            exp_logits = tl.exp(logits)
            # Mask out invalid positions by setting them to 0 for softmax (they were -inf -> exp=0)
            exp_logits = tl.where(mask_j, exp_logits, 0.0)
            softmax = exp_logits / tl.sum(exp_logits, axis=0)

            # Accumulate output: out[i, h, :] += softmax[j] * v[kv_start + j, orig_h, :]
            for j in tl.static_range(0, K_HEADS):
                orig_h = j
                pos = h * K_HEADS + j
                # Get softmax value for this expanded head
                sm = softmax[pos]
                # Load v[kv_start + j, orig_h, :]
                v_vec = tl.load(v_ptr + (kv_start + j) * (K_HEADS * D) + orig_h * D + tl.arange(0, D))
                # Accumulate
                out_vec = sm * v_vec
                # Store into out[i, h, :]
                tl.store(out_ptr + i * (Q_HEADS * D) + h * D + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure device is CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device for Triton kernel."

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

            Nq = q_end - q_start
            Nk = kv_end - kv_start

            # Launch Triton kernel for this segment
            attention_forward_segment[(1,)](
                q_batch, k_batch, v_batch, out, lse,
                q_start, kv_start, q_end, kv_end, sm_scale,
                Nq=Nq, Nk=Nk,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 as in the original spec
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
