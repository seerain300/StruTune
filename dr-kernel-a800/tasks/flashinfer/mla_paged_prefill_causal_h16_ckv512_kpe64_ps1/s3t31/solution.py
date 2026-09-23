import torch
import triton
import triton.language as tl
import math


# Kernel 1: For a given query i and head h, compute logits_scaled[h, :] of length L
# Inputs:
#   qn_vec_ptr: *fp32, length H*K (flatten q_nope[q_abs, :, :])
#   qp_vec_ptr: *fp32, length H*Kp (flatten q_pe[q_abs, :, :])
#   Kc_ptr: *fp32, base pointer to Kc_all (P*K)
#   Kp_ptr: *fp32, base pointer to Kp_all (P*Kp)
#   logits_scaled_ptr: *fp32, row buffer of length L for head h
#   tok_idx_ptr: *int32, length L token indices
#   H, K, Kp, L, sm_scale
#   head: constexpr, which head to compute
@triton.jit
def compute_logits_rows_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    tok_idx_ptr,
    H: tl.constexpr, K: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr,
    sm_scale: tl.constexpr,
    stride_qn_h: tl.constexpr, stride_qn_k: tl.constexpr,
    stride_qp_h: tl.constexpr, stride_qp_kp: tl.constexpr,
    stride_log_l: tl.constexpr,
    head: tl.constexpr,
):
    # Each program handles one head h
    # We will fill logits_scaled row for head h
    for l in range(L):
        tok = tl.load(tok_idx_ptr + l)  # int32 token index
        # Compute dot products
        acc = 0.0
        # Sum over K features for q_nope
        for k in range(K):
            base_qn = head * K + k
            qn_k = tl.load(qn_vec_ptr + base_qn)  # scalar fp32
            kc_k = tl.load(Kc_ptr + tok * K + k)  # scalar fp32
            acc += qn_k * kc_k
        # Sum over Kp features for q_pe
        for kp in range(Kp):
            base_qp = head * Kp + kp
            qp_kp = tl.load(qp_vec_ptr + base_qp)  # scalar fp32
            kp_kp = tl.load(Kp_ptr + tok * Kp + kp)  # scalar fp32
            acc += qp_kp * kp_kp
        # Scale
        acc = acc * sm_scale
        # Store
        tl.store(logits_scaled_ptr + l * stride_log_l, acc)


# Kernel 2: Compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2) after zeroing invalid positions
# Inputs:
#   logits_scaled_ptr: *fp32, row length L for head h
#   lse_ptr: *fp32, scalar pointer to lse[h]
#   L, valid_len (valid positions count)
@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L: tl.constexpr, valid_len: tl.constexpr,
    stride_log_l: tl.constexpr,
):
    # One program computes lse for a single head's row (lse_ptr points to scalar)
    # Zero invalid entries by reducing only valid positions
    sum_exp = 0.0
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        if l < valid_len:
            sum_exp += tl.exp(val)
    # Compute max of valid entries for numerical stability
    max_val = -1e30
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        if l < valid_len:
            max_val = tl.maximum(max_val, val)
    # Lse = log(sum(exp(val - max))) + max
    lse_val = tl.log(sum_exp) + max_val
    # Divide by ln(2)
    ln2 = 0.6931471805599453
    lse_val = lse_val / ln2
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute softmax over logits_scaled[h, :] and write attn[h, :]
# Inputs:
#   logits_scaled_ptr: *fp32, row length L
#   lse_ptr: *fp32, scalar pointer to lse[h]
#   attn_ptr: *fp32, row of length L for head h
#   L, valid_len
@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L: tl.constexpr, valid_len: tl.constexpr,
    stride_log_l: tl.constexpr,
    stride_attn_l: tl.constexpr,
):
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        if l < valid_len:
            attn_l = tl.exp(val - tl.load(lse_ptr))
        else:
            attn_l = 0.0
        tl.store(attn_ptr + l * stride_attn_l, attn_l)


# Kernel 4: GEMV: out[h, :] = attn[h, :] @ Kc_all[tok_idx[:], :]
# Inputs:
#   attn_ptr: *fp32, row of length L
#   Kc_ptr: *fp32, base pointer to Kc_all (P*K)
#   tok_idx_ptr: *int32, length L
#   out_vec_ptr: *fp32, output vector length K for head h
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, tok_idx_ptr, out_vec_ptr,
    L: tl.constexpr, K: tl.constexpr,
    stride_attn_l: tl.constexpr,
    stride_out_k: tl.constexpr,
):
    for k in range(K):
        acc = 0.0
        for l in range(L):
            attn_l = tl.load(attn_ptr + l * stride_attn_l)
            tok = tl.load(tok_idx_ptr + l)
            kc_k = tl.load(Kc_ptr + tok * K + k)
            acc += attn_l * kc_k
        tl.store(out_vec_ptr + k * stride_out_k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton kernels"
        device = q_nope.device

        # Squeeze caches to [P, K] and [P, Kp], cast to fp32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, K] with K=512
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Kp] with Kp=64

        total_q = q_nope.shape[0]
        H = q_nope.shape[1]  # num_qo_heads, assumed 16 in original
        K = Kc_all.shape[1]  # 512
        Kp = Kp_all.shape[1] # 64

        # Output and lse
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        # Process each batch element
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # Indices of tokens in this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L]

            # Iterate queries
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare qn_vec and qp_vec: flatten per-head vectors and cast to fp32
                qn = q_nope[q_abs]  # [H, K], bfloat16
                qn_vec = qn.view(-1).to(torch.float32).contiguous()  # [H*K]
                qp = q_pe[q_abs]    # [H, Kp], bfloat16
                qp_vec = qp.view(-1).to(torch.float32).contiguous()  # [H*Kp]

                # Allocate per-head buffers
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                attn = torch.empty((L,), device=device)

                # Compute logits_scaled row per head
                for h in range(H):
                    compute_logits_rows_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        tok_idx,
                        H=H, K=K, Kp=Kp, L=L, sm_scale=float(sm_scale),
                        stride_qn_h=K, stride_qn_k=1,
                        stride_qp_h=Kp, stride_qp_kp=1,
                        stride_log_l=1,
                        head=h,
                    )

                    # Determine valid_len for causal mask: positions j <= i are valid
                    # j maps to l, so valid positions are l >= q_len - i
                    valid_len = max(0, q_len - i)

                    # Compute lse
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse[q_abs, h],
                        L=L, valid_len=valid_len,
                        stride_log_l=1,
                    )

                    # Compute softmax
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse[q_abs, h], attn,
                        L=L, valid_len=valid_len,
                        stride_log_l=1, stride_attn_l=1,
                    )

                    # GEMV to get output vector for head h
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, tok_idx, out_vec,
                        L=L, K=K,
                        stride_attn_l=1, stride_out_k=1,
                    )

                    # Store output for this head as bfloat16
                    output[q_abs, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
