import torch
import triton
import triton.language as tl

# Constants
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


# Kernel: compute logits_scaled[h, l] for a single query i and head h
# qn_vec_ptr: *fp32, length H*K, flattened q_nope[q_abs] (without bias add)
# qp_vec_ptr: *fp32, length H*Kp, flattened q_pe[q_abs]
# Kc_ptr: *fp32, base pointer to Kc_all flattened [P*K]
# Kp_ptr: *fp32, base pointer to Kp_all flattened [P*Kp]
# tok_idx_ptr: *int32, length L, token indices into cache
# L: number of tokens in this batch segment
# H, K, Kp: dimensions (H=16, K=512, Kp=64)
# sm_scale: fp32 scalar
# head: which head to compute (constexpr)
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H, K, Kp, L,
    tok_idx_ptr,
    sm_scale,
    head: tl.constexpr,
):
    for l in range(0, L):
        idx = tl.load(tok_idx_ptr + l)
        acc_qn = 0.0
        acc_qp = 0.0
        # dot qn[h, :] with Kc[idx, :]
        for k in range(0, K):
            qn_val = tl.load(qn_vec_ptr + head * K + k)
            Kc_val = tl.load(Kc_ptr + idx * K + k)
            acc_qn += qn_val * Kc_val
        # dot qp[h, :] with Kp[idx, :]
        for kp in range(0, Kp):
            qp_val = tl.load(qp_vec_ptr + head * Kp + kp)
            Kp_val = tl.load(Kp_ptr + idx * Kp + kp)
            acc_qp += qp_val * Kp_val
        logits = acc_qn + acc_qp
        scaled = logits * sm_scale
        tl.store(logits_scaled_ptr + l, scaled)


# Kernel: compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2), zero invalid positions
# logits_scaled_ptr: *fp32, vector of length L
# lse_ptr: *fp32, scalar for this head
# L, prefix_len, i_query: see host for definitions
@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L,
    prefix_len,  # L - q_len
    i_query,     # current query i in this batch segment
):
    max_val = -1e30
    sum_exp = 0.0
    for j in range(0, L):
        if j > (prefix_len + i_query):
            val = -1e30
        else:
            val = tl.load(logits_scaled_ptr + j)
        if val > max_val:
            max_val = val
        # exclude invalid by setting val = -1e30 (exp->0)
        sum_exp += tl.exp(val - max_val)
    lse = tl.log(sum_exp) + max_val  # logsumexp over valid positions
    lse = lse / tl.log(2.0)          # convert to base-2
    tl.store(lse_ptr, lse)


# Kernel: compute attn[h, l] = exp(logits_scaled[h, l] - lse[h]) for valid positions, invalid set to 0
@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L,
    prefix_len,
    i_query,
):
    for j in range(0, L):
        if j > (prefix_len + i_query):
            attn_val = 0.0
        else:
            val = tl.load(logits_scaled_ptr + j)
            lse_val = tl.load(lse_ptr)
            attn_val = tl.exp(val - lse_val)
        tl.store(attn_ptr + j, attn_val)


# Kernel: GEMV out[h, :] = attn[h, :] @ Kc[tok_idx[:], :], produces output vector length K (512)
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, tok_idx_ptr, out_ptr,
    K, L,
):
    for k in range(0, K):
        acc = 0.0
        for l in range(0, L):
            attn_val = tl.load(attn_ptr + l)
            idx = tl.load(tok_idx_ptr + l)
            Kc_val = tl.load(Kc_ptr + idx * K + k)
            acc += attn_val * Kc_val
        tl.store(out_ptr + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        if q_nope.device.type != 'cuda':
            raise RuntimeError("ModelNew requires tensors on CUDA device.")
        device = q_nope.device

        # Cast caches to fp32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

        total_q = q_nope.shape[0]
        batch_size = qo_indptr.shape[0] - 1

        output = torch.empty(
            (total_q, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device
        )
        lse = torch.empty(
            (total_q, NUM_QO_HEADS), dtype=torch.float32, device=device
        )

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

            for i in range(q_len):
                q_abs = q_start + i

                # Flatten qn and qp: qn[h, k] -> vector [H*K], qp[h, kp] -> [H*Kp]
                qn = q_nope[q_abs]  # [16, 512]
                qn_vec = qn.contiguous().view(-1).to(torch.float32)  # [16*512]
                qp = q_pe[q_abs]    # [16, 64]
                qp_vec = qp.contiguous().view(-1).to(torch.float32)  # [16*64]

                # Allocate intermediate
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)

                # Compute logits for each head
                for h in range(NUM_QO_HEADS):
                    compute_logits_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        H=NUM_QO_HEADS, K=HEAD_DIM_CKV, Kp=HEAD_DIM_KPE, L=L,
                        tok_idx=tok_idx,
                        sm_scale=float(sm_scale),
                        head=h,
                    )
                    # lse
                    prefix_len = L - q_len
                    i_query = i
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse[q_abs, h],
                        L=L, prefix_len=prefix_len, i_query=i_query,
                    )
                    # attn
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse[q_abs, h], attn,
                        L=L, prefix_len=prefix_len, i_query=i_query,
                    )
                    # GEMV output
                    out_vec = torch.empty((HEAD_DIM_CKV,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, tok_idx, out_vec,
                        K=HEAD_DIM_CKV, L=L,
                    )
                    output[q_abs, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
