import torch
import triton
import triton.language as tl
import math


# Kernel: compute logits vector for one head h given flattened qn, qp and tok_idx
# qn_vec_ptr: *fp32, length H*K
# qp_vec_ptr: *fp32, length H*Kp
# Kc_ptr: *fp32, base pointer to Kc_all[P, K], we access Kc[tok_idx[l], k]
# Kp_ptr: *fp32, base pointer to Kp_all[P, Kp], we access Kp[tok_idx[l], kp]
# logits_scaled_ptr: *fp32, length L (row for head h)
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H, K, Kp, L,
    tok_idx_ptr,
    sm_scale: tl.float32,
    stride_qn_h: tl.constexpr, stride_qn_k: tl.constexpr,
    stride_qp_h: tl.constexpr, stride_qp_kp: tl.constexpr,
    stride_log_h: tl.constexpr, stride_log_l: tl.constexpr,
    head: tl.constexpr,
):
    # Compute logits[h, :] = sum_k qn[h, k] * Kc[tok_idx[l], k] + sum_kp qp[h, k'] * Kp[tok_idx[l], kp]
    for l in range(0, L):
        idx = tl.load(tok_idx_ptr + l)  # int32 index into caches
        base_kc = idx * K
        base_kp = idx * Kp
        acc = 0.0
        # Accumulate over K (head_dim_ckv)
        for k in range(0, K):
            qn_k = tl.load(qn_vec_ptr + head * K + k)
            kc = tl.load(Kc_ptr + base_kc + k)
            acc += qn_k * kc
        # Accumulate over Kp (head_dim_kpe)
        for kp in range(0, Kp):
            qp_kp = tl.load(qp_vec_ptr + head * Kp + kp)
            kp_val = tl.load(Kp_ptr + base_kp + kp)
            acc += qp_kp * kp_val
        acc *= sm_scale
        tl.store(logits_scaled_ptr + l, acc)


# Kernel: compute lse for one head h from logits_scaled (length L)
# lse[h] = log(sum(exp(logits_scaled[h, :]))) / ln(2)
@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L: tl.constexpr,
):
    sum_exp = 0.0
    for i in range(0, L):
        val = tl.load(logits_scaled_ptr + i)
        sum_exp += tl.exp(val)
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr, lse_val)


# Kernel: compute softmax for one head h using logits_scaled (length L) and lse[h]
# attn[h, l] = exp(logits_scaled[h, l] - lse[h]) for all l
@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L: tl.constexpr,
):
    lse_val = tl.load(lse_ptr)
    for i in range(0, L):
        val = tl.load(logits_scaled_ptr + i)
        attn_i = tl.exp(val - lse_val)
        tl.store(attn_ptr + i, attn_i)


# Kernel: perform GEMV for one head h: out[h, :] = attn[h, :] @ Kc_all[tok_idx[:], :]
# attn_ptr: *fp32, length L
# Kc_ptr: *fp32, base pointer to Kc_all[P, K]
# tok_idx_ptr: *int32, length L
# out_ptr: *fp32, length K
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, tok_idx_ptr, out_ptr,
    L, K,
    stride_out_h: tl.constexpr, stride_out_k: tl.constexpr,
):
    # For each output feature k, accumulate attn[l] * Kc[tok_idx[l], k]
    for k in range(0, K):
        dot = 0.0
        for l in range(0, L):
            attn_l = tl.load(attn_ptr + l)
            idx = tl.load(tok_idx_ptr + l)
            kc = tl.load(Kc_ptr + idx * K + k)  # Kc[tok_idx[l], k]
            dot += attn_l * kc
        tl.store(out_ptr + k, dot)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA tensors
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
    device = q_nope.device

    # Shapes
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    # Squeeze caches to [P, K] and [P, Kp]
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

    # Output and lse buffers
    output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    len_qo_indptr = qo_indptr.shape[0]
    for b in range(len_qo_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        q_len = q_end - q_start
        if q_len <= 0:
            continue

        # tokens in this kv segment
        tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).to(device)
        L = tok_idx.numel()

        # Process each query i in this batch element
        for i in range(q_len):
            q_abs = q_start + i

            # Flatten qn and qp
            qn = q_nope[q_abs]  # [H, K]
            qp = q_pe[q_abs]    # [H, Kp]
            qn_vec = qn.reshape(-1).contiguous()   # length H*K
            qp_vec = qp.reshape(-1).contiguous()   # length H*Kp

            # Intermediate buffers
            logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)

            # For each head
            for h in range(num_qo_heads):
                # Compute logits for this head
                compute_logits_kernel[(1,)](
                    qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                    H=num_qo_heads, K=head_dim_ckv, Kp=head_dim_kpe, L=L,
                    tok_idx=tok_idx,
                    sm_scale=float(sm_scale),
                    stride_qn_h=head_dim_ckv, stride_qn_k=1,
                    stride_qp_h=head_dim_kpe, stride_qp_kp=1,
                    stride_log_h=L, stride_log_l=1,
                    head=h,
                )

                # Compute lse for this head
                compute_lse_kernel[(1,)](
                    logits_scaled, lse[q_abs, h],
                    L=L,
                )

                # Compute attn for this head
                attn = torch.empty((L,), dtype=torch.float32, device=device)
                compute_softmax_kernel[(1,)](
                    logits_scaled, lse[q_abs, h], attn,
                    L=L,
                )

                # GEMV to produce output vector for this head
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                gemv_out_kernel[(1,)](
                    attn, Kc_all, tok_idx, out_vec,
                    L=L, K=head_dim_ckv,
                    stride_out_h=1, stride_out_k=1,
                )

                # Store into output
                output[q_abs, h, :] = out_vec

    # Match original return types: output bfloat16, lse float32
    return output.to(torch.bfloat16), lse


# Entry point for ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        if not q_nope.is_cuda:
            q_nope = q_nope.cuda()
        if not q_pe.is_cuda:
            q_pe = q_pe.cuda()
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.cuda()
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.cuda()
        if not qo_indptr.is_cuda:
            qo_indptr = qo_indptr.cuda()
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.cuda()
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.cuda()

        return run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
