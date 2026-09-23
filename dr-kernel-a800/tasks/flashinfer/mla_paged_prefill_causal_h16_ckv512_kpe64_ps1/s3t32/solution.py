import torch
import triton
import triton.language as tl
import math


@triton.jit
def compute_logits_rows_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    tok_idx_ptr,
    H: tl.constexpr, K: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr,
    sm_scale: tl.constexpr,
    stride_log_l: tl.constexpr,
    head: tl.constexpr,
):
    # For each token l, compute logits_scaled[head, l] = qn_vec[head*K:] @ Kc[tok_idx[l], :]
    # + qn_vec[head*Kp:] @ Kp[tok_idx[l], :], scaled by sm_scale
    for l in range(L):
        tok = tl.load(tok_idx_ptr + l)  # int32 token index
        acc = 0.0
        # Sum over K features from q_nope for this head
        for k in range(K):
            qn_k = tl.load(qn_vec_ptr + head * K + k)
            kc_k = tl.load(Kc_ptr + tok * K + k)
            acc += qn_k * kc_k
        # Sum over Kp features from q_pe for this head
        for kp in range(Kp):
            qp_kp = tl.load(qp_vec_ptr + head * Kp + kp)
            kp_kp = tl.load(Kp_ptr + tok * Kp + kp)
            acc += qp_kp * kp_kp
        acc = acc * sm_scale
        tl.store(logits_scaled_ptr + l * stride_log_l, acc)


@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L: tl.constexpr,
    stride_log_l: tl.constexpr,
):
    # Compute lse = logsumexp(logits_scaled[:]) / ln(2), ignoring invalid positions by masking them as non-positive
    sum_exp = 0.0
    max_val = -1e20
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        # If val > 0, consider it (host sets invalid positions <= 0)
        if val > 0:
            sum_exp += tl.exp(val)
    lse = tl.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr, lse)


@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L: tl.constexpr,
    stride_log_l: tl.constexpr,
):
    lse = tl.load(lse_ptr)
    # Softmax: attn[l] = exp(logits_scaled[l]) / lse if valid else 0 (invalid positions have logits_scaled <= 0 set by host)
    for l in range(L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        attn = tl.exp(val) / lse
        if val <= 0:
            attn = 0.0
        tl.store(attn_ptr + l * 1, attn)  # stride 1 assumed; we pass stride_log_l=1


@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    tok_idx_ptr,
    K: tl.constexpr, L: tl.constexpr,
    stride_attn_l: tl.constexpr,
    stride_out_k: tl.constexpr,
):
    # Compute out_vec[K] = attn[:] @ Kc[tok_idx[:], :]
    out_vec = [0.0] * K  # fp32 vector of size K
    for l in range(L):
        attn_l = tl.load(attn_ptr + l * stride_attn_l)
        # contribution from each feature k of Kc
        for k in range(K):
            tok = tl.load(tok_idx_ptr + l)
            kc_k = tl.load(Kc_ptr + tok * K + k)
            out_vec[k] += attn_l * kc_k
    # Store results
    for k in range(K):
        tl.store(out_ptr + k * stride_out_k, out_vec[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "cache has shape (num_pages, 1, ...)"

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        H = num_qo_heads
        K = head_dim_ckv
        Kp = head_dim_kpe

        ln2 = 1.0 / math.log(2.0)

        # Iterate batch elements
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            # Token indices for this batch element: tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b+1]]
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int32).to(device)
            L = tok_idx.shape[0]

            # Iterate queries
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare flattened q vectors (fp32 for Triton)
                qn = q_nope[q_abs]  # [H, K]
                qn_vec = qn.to(torch.float32).view(-1).contiguous()  # length H*K
                # For q_pe: [H, Kp]
                qp = q_pe[q_abs]  # [H, Kp]
                qp_vec = qp.to(torch.float32).view(-1).contiguous()  # length H*Kp

                # Allocate intermediates
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                attn = torch.empty((L,), dtype=torch.float32, device=device)

                # Compute logits rows per head
                for h in range(H):
                    # Launch compute_logits_rows_kernel
                    compute_logits_rows_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        tok_idx,
                        H=H, K=K, Kp=Kp, L=L,
                        sm_scale=float(sm_scale),
                        stride_log_l=1,
                        head=h,
                    )

                    # Mask invalid positions: for causal, position > prefix_len + i is invalid.
                    # prefix_len = number of previously cached tokens in this batch = L - q_len
                    prefix_len = L - q_len
                    # Host-side mask: set invalid logits to 0 (negative is fine for softmax), but we ensure they contribute 0 to lse
                    # We'll set to 0 explicitly for simplicity; softmax kernel will set attn=0 for <= 0.
                    invalid_start = prefix_len + 1 + i  # inclusive index where causal mask starts invalid
                    if invalid_start < L:
                        logits_scaled[invalid_start:] = 0.0

                    # Compute lse
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse[q_abs, h],
                        L=L,
                        stride_log_l=1,
                    )

                    # Compute softmax with invalid positions zeroed
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse[q_abs, h], attn,
                        L=L,
                        stride_log_l=1,
                    )

                    # GEMV: out[h, :] = attn[:] @ Kc_all[tok_idx[:], :]
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, out_vec,
                        tok_idx,
                        K=K, L=L,
                        stride_attn_l=1,
                        stride_out_k=1,
                    )

                    # Store result as bfloat16
                    output[q_abs, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
