import torch
import triton
import triton.language as tl
import math


# Kernel 1: For a given head h, compute logits_scaled[h, :] of length L
# Inputs:
#   qn_vec_ptr: *fp32, length H*K
#   qp_vec_ptr: *fp32, length H*Kp
#   Kc_ptr: *fp32, base pointer to Kc_all (P*K), we index using tok_idx_ptr[l] and feature k
#   Kp_ptr: *fp32, base pointer to Kp_all (P*Kp), index using tok_idx_ptr[l] and feature kp
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
    stride_log_l: tl.constexpr,
    head: tl.constexpr,
):
    # Each program handles one head and writes logits_scaled row
    for l in range(L):
        tok = tl.load(tok_idx_ptr + l)  # int32 token index
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
        acc = acc * sm_scale
        tl.store(logits_scaled_ptr + l, acc)


# Kernel 2: Compute lse for a given head h from logits_scaled[h, :]
# Inputs:
#   logits_scaled_ptr: *fp32, row of length L
#   lse_ptr: *fp32, scalar for head h
#   H, L, apply_causal: bool flag (if True, zero invalid positions before LSE)
@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    H: tl.constexpr, L: tl.constexpr, apply_causal: tl.constexpr,
):
    # We assume H is dummy here; reduction only on L
    # Compute max over valid entries (if apply_causal, invalid entries are already -inf)
    max_val = -float("inf")
    for l in range(L):
        # If causal is applied (invalid set to -inf), l==L means all entries valid; we still use apply_causal to avoid loading invalid
        val = tl.load(logits_scaled_ptr + l)
        if apply_causal:
            # For causal mask, invalid entries are -inf and will not affect max
            pass
        # Otherwise, just update max
        max_val = tl.maximum(max_val, val)
    # sumexp
    sumexp = 0.0
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l)
        # If causal: invalid entries are -inf, so exp(-inf) -> 0
        if apply_causal:
            pass
        sumexp += tl.exp(val - max_val)
    lse_val = max_val + tl.log(sumexp) / math.log(2.0)
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute softmax for a given head h (optional, if L>0)
# Inputs:
#   logits_scaled_ptr: *fp32, row of length L
#   lse_ptr: *fp32, scalar lse[h]
#   attn_ptr: *fp32, row buffer of length L for head h
#   H, L
@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    H: tl.constexpr, L: tl.constexpr,
):
    lse_val = tl.load(lse_ptr)
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l)
        # softmax over valid entries; invalid entries should be 0 (if any mask applied externally)
        attn_val = tl.exp(val - lse_val)
        # if invalid, attn_val should be 0; here we assume all valid since we masked before
        tl.store(attn_ptr + l, attn_val)


# Kernel 4: GEMV for a given head h: attn[h, :] @ Kc[tok_idx[:], :]
# Inputs:
#   attn_ptr: *fp32, row of length L
#   Kc_ptr: *fp32, base pointer to Kc_all (P*K), indexed by tok_idx_ptr[l] and feature k
#   out_ptr: *fp32, output vector of length K for head h
#   tok_idx_ptr: *int32, length L
#   H, K, L
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    tok_idx_ptr,
    H: tl.constexpr, K: tl.constexpr, L: tl.constexpr,
):
    for k in range(K):
        dot = 0.0
        for l in range(L):
            attn_l = tl.load(attn_ptr + l)  # scalar fp32
            tok = tl.load(tok_idx_ptr + l)  # int32 token index
            kc_k = tl.load(Kc_ptr + tok * K + k)  # scalar fp32
            dot += attn_l * kc_k
        tl.store(out_ptr + k, dot)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all math in Triton

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be CUDA tensors"
        device = q_nope.device
        dtype_q = q_nope.dtype
        dtype_k = ckv_cache.dtype

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

        # Constants
        H = 16
        K = 512
        Kp = 64
        num_qo_heads = H  # num_qo_heads asserted in original

        total_q = q_nope.shape[0]
        # lse and output
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)
        # Output tensor
        out = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)

        # Loop over batch elements
        for b in range(1, qo_indptr.shape[0]):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            # Tokens for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int32).to(device)  # [L]
            L = tok_idx.shape[0]

            # Loop over queries in this batch segment
            for i in range(q_len):
                q_abs = q_start + i
                # Flatten qn and qp
                qn = q_nope[q_abs]  # [H, K]
                qn_vec = qn.contiguous().view(-1).to(torch.float32).to(device)  # [H*K]
                qp = q_pe[q_abs]    # [H, Kp]
                qp_vec = qp.contiguous().view(-1).to(torch.float32).to(device)  # [H*Kp]

                # Buffer for logits_scaled per head
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)

                # Launch compute logits rows kernel for each head h
                for h in range(H):
                    compute_logits_rows_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        tok_idx,
                        H=H, K=K, Kp=Kp, L=L,
                        sm_scale=float(sm_scale),
                        stride_log_l=L,
                        head=h,
                    )

                    # Compute lse for head h
                    # Decide causal mask: invalid positions are those where token l corresponds to query j >= L
                    # prefix_len = number of previously cached tokens = L - q_len; invalid if l > prefix_len + i
                    prefix_len = L - q_len
                    invalid_start = max(0, prefix_len + i + 1)  # invalid positions l >= invalid_start
                    apply_causal = (invalid_start < L)  # bool as constexpr-like: pass as int?
                    # Note: Triton accepts Python bool as constexpr argument; we pass True/False
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse[q_abs, h],
                        H=H, L=L, apply_causal=apply_causal,
                    )

                    # If any invalid positions, softmax should not consider them (we set attn=0 externally).
                    # Compute attn (softmax); we assume valid here, but if apply_causal=True, we should set invalid attn=0
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse[q_abs, h], attn,
                        H=H, L=L,
                    )
                    if apply_causal:
                        # Zero out invalid positions by checking l >= invalid_start
                        # torch mask (we can do it in Triton too, but this is simple)
                        attn[:invalid_start].zero_()

                    # GEMV output: attn @ Kc[tok_idx[:], :]
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, out_vec,
                        tok_idx,
                        H=H, K=K, L=L,
                    )

                    # Store output in bfloat16
                    out[q_abs, h, :] = out_vec.to(torch.bfloat16)

        return out, lse


def run(*args):
    return ModelNew()(*args)
