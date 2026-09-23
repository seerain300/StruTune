import torch
import triton
import triton.language as tl

# Kernel: compute logits vector for one head h across L tokens
# qn_ptr: *fp32, points to q_nope[q_abs] flattened as [H*K]
# qp_ptr: *fp32, points to q_pe[q_abs] flattened as [H*Kp]
# Kc_ptr: *fp32, points to Kc_all flattened as [P*K]
# Kp_ptr: *fp32, points to Kp_all flattened as [P*Kp]
# logits_scaled_ptr: *fp32, output vector of length L for head h
# L: int, number of tokens in this batch segment
# sm_scale: fp32
@triton.jit
def compute_logits_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    L, sm_scale,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_log_l,
    head: tl.constexpr,
    K: tl.constexpr, Kp: tl.constexpr,
    tok_idx_ptr: tl.pointer_type(tl.int32),
):
    # This kernel is launched once per (b, i, h). It computes the entire logits_scaled vector for head h.
    # We'll loop over l in [0, L). Triton allows loops with runtime bounds.
    for l in range(0, L):
        # tok_idx = tok_idx_ptr[l]
        tok = tl.load(tok_idx_ptr + l)
        # Accumulate over K and Kp
        acc = 0.0
        # sum_k qn[h, k] * Kc[tok, k]
        for k in range(0, K):
            off_qn = head * K + k
            qn_val = tl.load(qn_ptr + off_qn)
            off_Kc = tok * K + k
            Kc_val = tl.load(Kc_ptr + off_Kc)
            acc += qn_val * Kc_val
        # sum_kp qp[h, k'] * Kp[tok, k']
        for kp in range(0, Kp):
            off_qp = head * Kp + kp
            qp_val = tl.load(qp_ptr + off_qp)
            off_Kp = tok * Kp + kp
            Kp_val = tl.load(Kp_ptr + off_Kp)
            acc += qp_val * Kp_val
        # scale
        acc = acc * sm_scale
        tl.store(logits_scaled_ptr + l * stride_log_l, acc)


# Kernel: compute lse for one head from logits_scaled vector of length L
# logits_scaled_ptr: *fp32, vector of length L
# lse_ptr: *fp32, scalar output
@triton.jit
def lse_kernel(logits_scaled_ptr, lse_ptr, L, stride_log_l):
    # Compute max
    maxv = -float('inf')
    for l in range(0, L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        if val > maxv:
            maxv = val
    # Compute sum exp
    sumexp = 0.0
    for l in range(0, L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        sumexp += tl.exp(val - maxv)
    lse = tl.log(sumexp) + maxv  # logsumexp
    # Divide by ln(2)
    ln2 = 0.6931471805599453
    lse = lse / ln2
    tl.store(lse_ptr, lse)


# Kernel: compute softmax for one head from logits_scaled and lse, write attn
# logits_scaled_ptr: *fp32, vector of length L
# lse_ptr: *fp32, scalar
# attn_ptr: *fp32, output vector of length L
@triton.jit
def softmax_kernel(logits_scaled_ptr, lse_ptr, attn_ptr, L, stride_log_l, stride_attn_l):
    # Load lse
    lse = tl.load(lse_ptr)
    for l in range(0, L):
        val = tl.load(logits_scaled_ptr + l * stride_log_l)
        attn_val = tl.exp(val - lse)  # softmax value
        tl.store(attn_ptr + l * stride_attn_l, attn_val)


# Kernel: GEMV for one head: out[h, :] = attn[h, :] @ Kc[tok_idx[:], :]
# Kc_ptr: *fp32, points to Kc_all flattened [P*K]
# tok_idx_ptr: *int32, length L
# attn_ptr: *fp32, points to attn[h, :] flattened vector of length L
# out_ptr: *fp32, output vector of length K
@triton.jit
def gemv_kernel(Kc_ptr, tok_idx_ptr, attn_ptr, out_ptr, K, L, stride_out_k, stride_attn_l):
    for k in range(0, K):
        acc = 0.0
        for l in range(0, L):
            tok = tl.load(tok_idx_ptr + l)
            attn_val = tl.load(attn_ptr + l * stride_attn_l)
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            acc += attn_val * Kc_val
        tl.store(out_ptr + k * stride_out_k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        head_dim_kpe = q_pe.shape[-1]
        assert head_dim_kpe == 64
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
        batch_size = qo_indptr.shape[0] - 1
        num_kv_indices = kv_indices.shape[0]

        # Prepare caches on device
        Kc_all = ckv_cache.squeeze(1).to(device=device, dtype=torch.float32).contiguous()  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(device=device, dtype=torch.float32).contiguous()  # [P, 64]

        output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Loop over batch elements
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len <= 0:
                continue

            # Compute tok_idx for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue
            tok_idx = kv_indices[page_beg:page_end].to(device=device, dtype=torch.int32).contiguous()  # [L]
            L = tok_idx.shape[0]

            # Process each query index i
            for i in range(q_len):
                q_abs = q_start + i
                # Prepare qn_vec and qp_vec as float32, shape [H*K] and [H*Kp], contiguous
                qn = q_nope[q_abs].to(device=device, dtype=torch.float32).contiguous()  # [16, 512]
                qn_vec = qn.view(-1).contiguous()  # [H*K]
                qp = q_pe[q_abs].to(device=device, dtype=torch.float32).contiguous()   # [16, 64]
                qp_vec = qp.view(-1).contiguous()  # [H*Kp]

                # For each head, compute logits_scaled, lse, attn, and output
                for h in range(16):
                    # logits_scaled: [L] float32
                    logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                    stride_log_l = 1

                    # Compute logits for head h
                    compute_logits_row_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        L=L, sm_scale=float(sm_scale),
                        stride_qn_h=512, stride_qn_k=1,
                        stride_qp_h=64, stride_qp_kp=1,
                        stride_log_l=stride_log_l,
                        head=h, K=512, Kp=64, tok_idx_ptr=tok_idx,
                    )

                    # Compute lse for head h
                    lse_b_h = torch.empty((1,), dtype=torch.float32, device=device)
                    lse_kernel[(1,)](
                        logits_scaled, lse_b_h, L=L, stride_log_l=stride_log_l
                    )
                    lse[q_abs, h] = lse_b_h[0]

                    # Compute attn for head h
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    stride_attn_l = 1
                    softmax_kernel[(1,)](
                        logits_scaled, lse[q_abs, h], attn, L=L, stride_log_l=stride_log_l, stride_attn_l=stride_attn_l
                    )

                    # GEMV: out[h, :] = attn @ Kc[tok_idx[:], :]
                    out_vec = torch.empty((512,), dtype=torch.float32, device=device)
                    stride_out_k = 1
                    gemv_kernel[(1,)](
                        Kc_all, tok_idx, attn, out_vec, K=512, L=L, stride_out_k=stride_out_k, stride_attn_l=stride_attn_l
                    )

                    # Store output in [q_abs, h, :]
                    output[q_abs, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
