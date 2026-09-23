import math
import torch
import triton
import triton.language as tl


# Kernel: compute logits_scaled per head h for a given query i
# Arguments:
#   qn_vec_ptr: *fp32, length H*K (H=16, K=512)
#   qp_vec_ptr: *fp32, length H*Kp (H=16, Kp=64)
#   Kc_ptr: *fp32, base pointer to Kc_all (P*K)
#   Kp_ptr: *fp32, base pointer to Kp_all (P*Kp)
#   tok_idx_ptr: *int32, length L
#   L: int32
#   H: int32 (compile-time in kernel signature)
#   K: int32
#   Kp: int32
#   sm_scale: fp32
#   logits_scaled_ptr: *fp32, output row for head h, length L
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H: tl.constexpr, K: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr,
    tok_idx_ptr, sm_scale: tl.float32
):
    h = tl.program_id(0)  # one program per head
    for l in tl.static_range(0, L):
        tok = tl.load(tok_idx_ptr + l)  # int32 token index
        sum_qn = 0.0
        # k loop over 0..K-1
        for k in tl.static_range(0, K):
            qn_k = tl.load(qn_vec_ptr + h * K + k)  # qn_vec laid out as h-major
            kc = tl.load(Kc_ptr + tok * K + k)
            sum_qn += qn_k * kc
        sum_qp = 0.0
        for kp in tl.static_range(0, Kp):
            qp_kp = tl.load(qp_vec_ptr + h * Kp + kp)  # qp_vec laid out as h-major
            kpval = tl.load(Kp_ptr + tok * Kp + kp)
            sum_qp += qp_kp * kpval
        logits_scaled = sum_qn + sum_qp
        logits_scaled *= sm_scale
        tl.store(logits_scaled_ptr + l, logits_scaled)


# Kernel: compute lse per head (rowwise logsumexp on logits_scaled)
# Arguments:
#   logits_scaled_ptr: *fp32, length L
#   lse_ptr: *fp32, scalar output for lse[h]
#   L: int32
@triton.jit
def compute_lse_kernel(logits_scaled_ptr, lse_ptr, L: tl.constexpr):
    # Compute max for stability
    max_val = -1.0e30
    for l in tl.static_range(0, L):
        max_val = tl.maximum(max_val, tl.load(logits_scaled_ptr + l))
    sum_exp = 0.0
    for l in tl.static_range(0, L):
        sum_exp += tl.exp(tl.load(logits_scaled_ptr + l) - max_val)
    lse = tl.log(sum_exp) + max_val
    ln2 = 0.6931471805599453  # math.log(2.0)
    tl.store(lse_ptr, lse / ln2)


# Kernel: compute softmax per head: attn[l] = exp(logits_scaled[l] - lse)
# Arguments:
#   logits_scaled_ptr: *fp32, length L
#   lse_ptr: *fp32, scalar lse for head
#   attn_ptr: *fp32, length L
@triton.jit
def compute_softmax_kernel(logits_scaled_ptr, lse_ptr, attn_ptr, L: tl.constexpr):
    lse = tl.load(lse_ptr)
    for l in tl.static_range(0, L):
        x = tl.load(logits_scaled_ptr + l)
        attn_l = tl.exp(x - lse)  # no ln2 here; lse already scaled in host if needed
        tl.store(attn_ptr + l, attn_l)


# Kernel: GEMV out per head: out_vec[k] = sum_l attn[l] * Kc[tok_idx[l], k]
# Arguments:
#   attn_ptr: *fp32, length L
#   Kc_ptr: *fp32, base pointer to Kc_all (P*K)
#   tok_idx_ptr: *int32, length L
#   out_vec_ptr: *fp32, length K (for head h)
@triton.jit
def gemv_out_kernel(attn_ptr, Kc_ptr, tok_idx_ptr, out_vec_ptr, L: tl.constexpr, K: tl.constexpr):
    for k in tl.static_range(0, K):
        acc = 0.0
        for l in tl.static_range(0, L):
            attn_l = tl.load(attn_ptr + l)
            tok = tl.load(tok_idx_ptr + l)
            kc_k = tl.load(Kc_ptr + tok * K + k)
            acc += attn_l * kc_k
        tl.store(out_vec_ptr + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtype for compute
        device = q_nope.device
        # Squeeze caches to [P, dim]
        ckv_cache = ckv_cache.squeeze(1).to(device=device, dtype=torch.float32)
        kpe_cache = kpe_cache.squeeze(1).to(device=device, dtype=torch.float32)

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        H = num_qo_heads
        K = head_dim_ckv
        Kp = head_dim_kpe

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            # token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).to(device)
            L = tok_idx.shape[0]

            # absolute query index within global q_nope
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare flattened vectors of q_nope and q_pe for head-wise dot products (float32 compute)
                qn = q_nope[q_abs]  # [16, 512]
                qn_vec = qn.contiguous().view(-1).to(torch.float32).to(device)  # [H*K]
                qp = q_pe[q_abs]    # [16, 64]
                qp_vec = qp.contiguous().view(-1).to(torch.float32).to(device)  # [H*Kp]

                # Allocate intermediates
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                lse_h = torch.empty((), dtype=torch.float32, device=device)
                attn = torch.empty((L,), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for each head h
                for h in range(H):
                    compute_logits_kernel[(1,)](
                        qn_vec, qp_vec, ckv_cache, kpe_cache, logits_scaled,
                        H=H, K=K, Kp=Kp, L=L, tok_idx=tok_idx, sm_scale=float(sm_scale),
                        num_warps=1, num_stages=1
                    )

                    # Compute lse for head h
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse_h, L=L
                    )

                    # Compute attn
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse_h, attn, L=L
                    )

                    # GEMV to produce output[h, :]
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, ckv_cache, tok_idx, out_vec, L=L, K=K,
                        num_warps=1, num_stages=1
                    )

                    # Store output (cast to bfloat16)
                    output[q_abs, h] = out_vec.to(torch.bfloat16)
                    # Store lse per head (scalar)
                    lse[q_abs, h] = lse_h

        return output, lse


def run(*args):
    return ModelNew()(*args)
