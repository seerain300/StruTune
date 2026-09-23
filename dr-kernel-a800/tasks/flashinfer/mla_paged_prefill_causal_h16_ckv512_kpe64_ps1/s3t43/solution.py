import torch
import triton
import triton.language as tl

# Constants (from assertions in original code)
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64

# Kernel 1: Compute per-head logits for tokens j in [0..L-1]
# qn_vec_ptr: *fp32, length H*K, q_nope[q_abs] flattened
# qp_vec_ptr: *fp32, length H*Kp, q_pe[q_abs] flattened
# Kc_ptr: *fp32, base pointer to Kc_all flattened (P*K)
# Kp_ptr: *fp32, base pointer to Kp_all flattened (P*Kp)
# tok_idx_ptr: *int32, length L
# L: number of tokens in this batch segment
# H, K, Kp: dimensions
# sm_scale: fp32 scalar
# out_ptr: *fp32, contiguous buffer of size H*L (one per head)
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, out_ptr,
    H, K, Kp, L,
    tok_idx_ptr,
    sm_scale,
):
    # program id: one program per head h
    h = tl.program_id(0)
    base = h * L
    for j in range(0, L):
        acc = 0.0
        # accumulate qn contribution: sum_k qn[h, k] * Kc[tok_idx[j], k]
        for k in range(0, K):
            qn_val = tl.load(qn_vec_ptr + h * K + k)
            idx = tl.load(tok_idx_ptr + j)
            kc_val = tl.load(Kc_ptr + idx * K + k)
            acc += qn_val * kc_val
        # accumulate qp contribution: sum_kp qp[h, kp] * Kp[tok_idx[j], kp]
        for kp in range(0, Kp):
            qp_val = tl.load(qp_vec_ptr + h * Kp + kp)
            idx = tl.load(tok_idx_ptr + j)
            kp_val = tl.load(Kp_ptr + idx * Kp + kp)
            acc += qp_val * kp_val
        acc *= sm_scale
        tl.store(out_ptr + base + j, acc)


# Kernel 2: Compute logsumexp over a vector (scaled logits), returns scalar in lse_ptr
# log_ptr: *fp32, input vector of length L (logits_scaled)
# L: int
# lse_ptr: *fp32, scalar output
@triton.jit
def compute_lse_kernel(log_ptr, lse_ptr, L):
    # Compute max
    m = -1e20
    for j in range(0, L):
        v = tl.load(log_ptr + j)
        if v > m:
            m = v
    # Compute sum exp(v - m)
    s = 0.0
    for j in range(0, L):
        v = tl.load(log_ptr + j)
        s += tl.exp(v - m)
    # lse = m + log(s)
    lse_val = m + tl.log(s)
    # divide by ln(2)
    lse_val *= 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute softmax over a vector (scaled logits) and store attn in attn_ptr
# log_ptr: *fp32, input vector of length L (logits_scaled)
# lse_ptr: *fp32, scalar lse (per head)
# attn_ptr: *fp32, output vector of length L
@triton.jit
def compute_softmax_kernel(log_ptr, lse_ptr, attn_ptr, L):
    lse_val = tl.load(lse_ptr)
    for j in range(0, L):
        v = tl.load(log_ptr + j)
        a = tl.exp(v - lse_val)
        tl.store(attn_ptr + j, a)


# Kernel 4: GEMV for attn[h, :] @ Kc[tok_idx[:], :] → out_vec_ptr[h*K + k]
# attn_ptr: *fp32, vector of length L
# Kc_ptr: *fp32, base pointer to Kc_all flattened (P*K)
# tok_idx_ptr: *int32, length L
# K: int
# out_vec_ptr: *fp32, vector of length K, corresponds to output[h, :]
@triton.jit
def gemv_out_kernel(attn_ptr, Kc_ptr, tok_idx_ptr, out_vec_ptr, L, K):
    h = tl.program_id(0)
    for k in range(0, K):
        acc = 0.0
        for l in range(0, L):
            attn_val = tl.load(attn_ptr + l)
            idx = tl.load(tok_idx_ptr + l)
            Kc_val = tl.load(Kc_ptr + idx * K + k)
            acc += attn_val * Kc_val
        tl.store(out_vec_ptr + h * K + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA
        if q_nope.device.type != 'cuda':
            raise RuntimeError("ModelNew requires tensors on CUDA device.")
        device = q_nope.device

        # Cast caches to fp32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

        total_q = q_nope.shape[0]
        batch_size = qo_indptr.shape[0] - 1

        # Output tensors (final stored as bfloat16)
        output = torch.empty((total_q, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, NUM_QO_HEADS), dtype=torch.float32, device=device)

        H = NUM_QO_HEADS
        K = HEAD_DIM_CKV
        Kp = HEAD_DIM_KPE

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L = end - start
            tok_idx = kv_indices[start:end].to(torch.int32).to(device).contiguous()  # [L]

            # Per query i
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare qn_vec_flat and qp_vec_flat (flattened across heads)
                qn_base = q_nope[q_abs].to(torch.float32)  # [H, K]
                qn_vec_flat = qn_base.view(-1)            # [H*K]
                qp_base = q_pe[q_abs].to(torch.float32)  # [H, Kp]
                qp_vec_flat = qp_base.view(-1)           # [H*Kp]

                # Buffer for logits_scaled[h, :] across heads (size H*L)
                logits_buf = torch.empty((H * L,), dtype=torch.float32, device=device)

                # Kernel: compute logits_scaled for all heads
                for h in range(H):
                    base = h * L
                    compute_logits_kernel[(1,)](
                        qn_vec_flat, qp_vec_flat, Kc_all, Kp_all,
                        logits_buf + base,  # out_ptr points to this head's row
                        H=H, K=K, Kp=Kp, L=L,
                        tok_idx_ptr=tok_idx,
                        sm_scale=float(sm_scale),
                    )

                    # Kernel: compute lse for this head
                    lse_scalar = torch.empty((), dtype=torch.float32, device=device)
                    compute_lse_kernel[(1,)](
                        logits_buf + base, lse_scalar, L
                    )
                    # Store lse for this query and head
                    lse[q_abs, h] = lse_scalar.item()

                    # Kernel: compute attn for this head
                    attn_vec = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_kernel[(1,)](
                        logits_buf + base, lse_scalar, attn_vec, L
                    )

                    # Kernel: GEMV to compute output[h, :]
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn_vec, Kc_all, tok_idx, out_vec, L, K
                    )

                    # Store output as bfloat16
                    output[q_abs, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
