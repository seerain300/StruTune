import torch
import triton
import triton.language as tl


# Kernel 1: Compute per-head logits for tokens [0..L-1]
# We run this kernel once per (b, i, h). It writes logits_scaled[h, :] into a provided buffer.
# Args:
#   qn_vec_ptr: *fp32, length H*K, flattened q_nope[q_abs]
#   qp_vec_ptr: *fp32, length H*Kp, flattened q_pe[q_abs]
#   Kc_ptr: *fp32, base pointer to Kc_all flattened (P*K)
#   Kp_ptr: *fp32, base pointer to Kp_all flattened (P*Kp)
#   logits_scaled_ptr: *fp32, buffer to store logits_scaled[h, :] of length L
#   H, K, Kp, L: dimensions
#   tok_idx_ptr: *int32, length L
#   sm_scale: fp32 scalar
#   head: constexpr, head index
@triton.jit
def compute_logits_per_head_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H, K, Kp, L,
    tok_idx_ptr,
    sm_scale,
    head: tl.constexpr,
):
    # For each token l, compute logits[h, l] = sum_k qn[h, k]*Kc[tok_idx[l], k] + sum_kp qp[h, k'] * Kp[tok_idx[l], k']
    l = 0
    while l < L:
        # Load token index
        idx_l = tl.load(tok_idx_ptr + l)  # int32
        # Accumulate over K and Kp
        sum_qn = tl.zeros((), dtype=tl.float32)
        k = 0
        while k < K:
            qn_val = tl.load(qn_vec_ptr + head * K + k)  # qn[h, k]
            kc_val = tl.load(Kc_ptr + idx_l * K + k)     # Kc[tok_idx[l], k]
            sum_qn += qn_val * kc_val
            k += 1

        sum_qp = tl.zeros((), dtype=tl.float32)
        kp = 0
        while kp < Kp:
            qp_val = tl.load(qp_vec_ptr + head * Kp + kp)  # qp[h, kp]
            kp_val = tl.load(Kp_ptr + idx_l * Kp + kp)     # Kp[tok_idx[l], kp]
            sum_qp += qp_val * kp_val
            kp += 1

        logits_scaled = sum_qn + sum_qp
        logits_scaled = logits_scaled * sm_scale
        tl.store(logits_scaled_ptr + l, logits_scaled)
        l += 1


# Kernel 2: Compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
# This kernel assumes logits_scaled_ptr points to a single row of length L.
@triton.jit
def compute_lse_row_kernel(
    logits_scaled_ptr, lse_ptr,
    L,
    head: tl.constexpr,
):
    # Row-wise max
    max_val = tl.full((), -float("inf"), dtype=tl.float32)
    l = 0
    while l < L:
        val = tl.load(logits_scaled_ptr + l)
        max_val = tl.maximum(max_val, val)
        l += 1

    # Row-wise sum of exp(logits - max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    l = 0
    while l < L:
        val = tl.load(logits_scaled_ptr + l)
        sum_exp += tl.exp(val - max_val)
        l += 1

    lse_val = max_val + tl.log(sum_exp)
    # 1 / ln(2)
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr + head, lse_val * inv_ln2)


# Kernel 3: Compute softmax per head on logits_scaled[h, :] with invalid positions zeroed
# We zero-out positions where j > prefix_len + i (prefix_len = L - q_len).
# Args:
#   logits_scaled_ptr: *fp32, length L
#   lse_ptr: *fp32, scalar lse[h]
#   attn_ptr: *fp32, length L to store softmax values
#   L, prefix_len: int scalars
#   head: constexpr
@triton.jit
def compute_softmax_row_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L, prefix_len,
    head: tl.constexpr,
):
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse_val = tl.load(lse_ptr + head)  # lse[h] = logsumexp * inv_ln2
    # We need to recover logsumexp[h] = lse_val / inv_ln2
    # But softmax uses lse_val directly: attn = exp(val - lse_val) for valid; else 0
    l = 0
    while l < L:
        j = l  # position index
        is_valid = j <= prefix_len
        val = tl.load(logits_scaled_ptr + l)
        soft = tl.exp(val - lse_val) if is_valid else 0.0
        tl.store(attn_ptr + l, soft)
        l += 1


# Kernel 4: GEMV out[h, :] = attn[h, :] @ Kc_all[tok_idx[:], :]
# We compute out[h, k] = sum_l attn[h, l] * Kc_all[tok_idx[l], k] for k in [0..K-1]
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H, L, K,
    tok_idx_ptr,  # *int32, length L
    head: tl.constexpr,
):
    k = 0
    acc = tl.zeros((), dtype=tl.float32)
    while k < K:
        l = 0
        while l < L:
            idx_l = tl.load(tok_idx_ptr + l)
            attn_val = tl.load(attn_ptr + l)
            kc_val = tl.load(Kc_ptr + idx_l * K + k)
            acc += attn_val * kc_val
            l += 1
        # Store acc to out[h, k]
        tl.store(out_ptr + head * K + k, acc)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure we are on CUDA
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        # Squeeze caches: [P, 1, ...] -> [P, ...]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, 64]

        H = 16
        K = 512
        Kp = 64

        total_q = int(qo_indptr[-1].item())
        num_batches = qo_indptr.numel() - 1
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        # Iterate over batch elements
        for b in range(num_batches):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            # Compute token indices for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L <= 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L]
            # Loop over queries i in this batch
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare qn_vec and qp_vec: flattened vectors
                qn = q_nope[q_abs]  # [H, K]
                qn_vec = qn.contiguous().view(-1).to(torch.float32)  # [H*K]
                qp = q_pe[q_abs]    # [H, Kp]
                qp_vec = qp.contiguous().view(-1).to(torch.float32)  # [H*Kp]

                # Allocate row buffers
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                attn = torch.empty((L,), dtype=torch.float32, device=device)

                # 1) Compute logits_scaled per head: we need a 2D launch or loop over heads.
                # However, Triton doesn't support returning 2D from a single launch easily; we'll loop heads and write per-head.
                for h in range(H):
                    compute_logits_per_head_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        H=H, K=K, Kp=Kp, L=L, tok_idx=tok_idx, sm_scale=float(sm_scale),
                        head=h,
                    )

                    # 2) Compute lse[h] = logsumexp(logits_scaled) / ln(2)
                    compute_lse_row_kernel[(1,)](
                        logits_scaled, lse[q_abs],  # write to lse[q_abs, h]
                        L=L, head=h,
                    )

                    # 3) Compute attn[h, :] = softmax(logits_scaled) with causal mask
                    prefix_len = (L - q_len)  # original code uses kv_len - q_len; here L is current tokens, q_len queries in this batch
                    compute_softmax_row_kernel[(1,)](
                        logits_scaled, lse[q_abs], attn,
                        L=L, prefix_len=prefix_len, head=h,
                    )

                    # 4) Compute out[h, :] = attn[h, :] @ Kc_all[tok_idx[:], :]
                    out_row = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, out_row,
                        H=H, L=L, K=K, tok_idx=tok_idx, head=h,
                    )

                    # Store output as bfloat16
                    output[q_abs] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
