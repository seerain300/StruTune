import math
import torch
import triton
import triton.language as tl


# Kernel 1: compute_logits_kernel
# For a given head h and each token l in [0..L), compute:
#   logits[h, l] = sum_k qn[h, k] * Kc[tok_idx[l], k] + sum_kp qp[h, kp] * Kp[tok_idx[l], kp]
# Then store logits_scaled[h, l] = logits[h, l] * sm_scale
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, out_ptr,
    H: tl.constexpr, K: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr,
    tok_idx_ptr,
    sm_scale: tl.constexpr,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_out_h, stride_out_l,
    head: tl.constexpr,
):
    for l in range(L):
        acc = 0.0
        # qn contribution
        for k in range(K):
            qn_val = tl.load(qn_vec_ptr + head * stride_qn_h + k * stride_qn_k)
            tok_l = tl.load(tok_idx_ptr + l)
            kc_val = tl.load(Kc_ptr + tok_l * K + k)  # Kc_ptr is [P, K] flattened
            acc += qn_val * kc_val
        # qp contribution
        for kp in range(Kp):
            qp_val = tl.load(qp_vec_ptr + head * stride_qp_h + kp * stride_qp_kp)
            tok_l = tl.load(tok_idx_ptr + l)
            kp_val = tl.load(Kp_ptr + tok_l * Kp + kp)
            acc += qp_val * kp_val
        acc = acc * sm_scale
        tl.store(out_ptr + head * stride_out_h + l * stride_out_l, acc)


# Kernel 2: compute_lse_kernel
# Given logits_scaled_ptr as [H, L] (row-major), compute lse[h] = logsumexp(logits_scaled[h, :]) / LN2.
# We avoid -inf: compute max, sum(exp(x - max)), then lse = log(sum) / LN2.
@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    H: tl.constexpr, L: tl.constexpr,
    stride_log_h, stride_log_l,
    LN2: tl.constexpr,
):
    h = tl.program_id(0)
    # Load vector for head h
    vec = [tl.load(logits_scaled_ptr + h * stride_log_h + l * stride_log_l) for l in range(L)]
    max_val = -float('inf')
    for v in vec:
        if v > max_val:
            max_val = v
    sum_exp = 0.0
    for v in vec:
        sum_exp += tl.exp(v - max_val)
    lse_val = tl.log(sum_exp) / LN2
    tl.store(lse_ptr + h, lse_val)


# Kernel 3: compute_softmax_kernel
# Given logits_scaled_ptr as [H, L] and lse[h] (scalar), compute attn[h, l] = exp((logits_scaled[h, l] - lse[h])).
@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    H: tl.constexpr, L: tl.constexpr,
    stride_log_h, stride_log_l,
    stride_attn_h, stride_attn_l,
    LN2: tl.constexpr,  # not used here, kept for signature symmetry
):
    h = tl.program_id(0)
    lse_val = tl.load(lse_ptr + h)
    for l in range(L):
        v = tl.load(logits_scaled_ptr + h * stride_log_h + l * stride_log_l)
        attn_val = tl.exp(v - lse_val)
        tl.store(attn_ptr + h * stride_attn_h + l * stride_attn_l, attn_val)


# Kernel 4: GEMV kernel to compute output[h, :] = attn[h, :] @ Kc[tok_idx, :]
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, tok_idx_ptr, out_ptr,
    H: tl.constexpr, K: tl.constexpr, L: tl.constexpr,
    stride_attn_h, stride_attn_l,
    stride_out_h, stride_out_k,
):
    h = tl.program_id(0)
    for k in range(K):
        acc = 0.0
        for l in range(L):
            attn_l = tl.load(attn_ptr + h * stride_attn_h + l * stride_attn_l)
            tok_l = tl.load(tok_idx_ptr + l)
            kc_val = tl.load(Kc_ptr + tok_l * K + k)
            acc += attn_l * kc_val
        tl.store(out_ptr + h * stride_out_h + k * stride_out_k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, K]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Kp]
        K = Kc_all.shape[-1]
        Kp = Kp_all.shape[-1]
        assert K == 512, "K must be 512"
        assert Kp == 64, "Kp must be 64"

        total_q = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"

        # Cast index tensors to int32
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)

        # Outputs
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        LN2 = math.log(2.0)

        batch_size = qo_indptr.shape[0] - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            q_len = q_end - q_start
            L = kv_end - kv_start

            if q_len <= 0 or L <= 0:
                continue

            # Token indices for this batch element: int32
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)  # [L]

            # For each query i
            for i in range(q_len):
                q_abs = q_start + i

                # Build flattened vectors for Triton: qn_vec_ptr [H*K], qp_vec_ptr [H*Kp]
                # q_nope[q_abs] shape [H, K]; q_pe[q_abs] shape [H, Kp]
                qn = q_nope[q_abs]  # [H, K], fp16 or fp32; we read as fp32 below
                qp = q_pe[q_abs]    # [H, Kp]
                # Create fp32 flat vectors for Triton kernels
                qn_vec = qn.view(-1).contiguous().to(torch.float32)  # [H*K]
                qp_vec = qp.view(-1).contiguous().to(torch.float32)  # [H*Kp]

                # Buffers for logits_scaled [H, L], attn [H, L]
                logits_scaled = torch.empty((H, L), dtype=torch.float32, device=device)
                attn = torch.empty((H, L), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for each head h
                for h in range(H):
                    compute_logits_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled[h, :],
                        H=H, K=K, Kp=Kp, L=L, tok_idx_ptr=tok_idx,
                        sm_scale=float(sm_scale),
                        stride_qn_h=K, stride_qn_k=1,
                        stride_qp_h=Kp, stride_qp_kp=1,
                        stride_out_h=1, stride_out_l=1,
                        head=h,
                    )

                    # Compute lse[h] = logsumexp(logits_scaled[h, :]) / LN2
                    compute_lse_kernel[(1,)](
                        logits_scaled[h, :], lse[q_abs, h],
                        H=1, L=L,
                        stride_log_h=1, stride_log_l=1,
                        LN2=float(LN2),
                    )

                    # Compute attn[h, :] = softmax(logits_scaled[h, :] - lse[h])
                    compute_softmax_kernel[(1,)](
                        logits_scaled[h, :], lse[q_abs, h], attn[h, :],
                        H=1, L=L,
                        stride_log_h=1, stride_log_l=1,
                        stride_attn_h=1, stride_attn_l=1,
                        LN2=float(LN2),
                    )

                    # Compute output[h, :] = attn[h, :] @ Kc_all[tok_idx, :] (GEMV), vector of length K
                    out_row = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn[h, :], Kc_all, tok_idx, out_row,
                        H=1, K=K, L=L,
                        stride_attn_h=1, stride_attn_l=1,
                        stride_out_h=1, stride_out_k=1,
                    )
                    output[q_abs, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
