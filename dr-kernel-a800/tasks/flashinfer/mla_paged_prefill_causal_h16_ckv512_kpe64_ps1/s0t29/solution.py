import torch
import math

import triton
import triton.language as tl


# GEMV kernel for D_in=512: y[M] = q_row[512] @ Kc_rows[M, 512].T
@triton.jit
def row_gemv_512_kernel(q_ptr,  # *float32, q_row of length 512
                        K_ptr,  # *float32, K_rows[M, 512] contiguous
                        y_ptr,  # *float32, output vector [M]
                        M: tl.constexpr):  # number of rows in K
    pid_m = tl.program_id(0)  # each program handles one output element along M
    acc = tl.zeros([1], dtype=tl.float32)
    # Iterate over feature dimension 512
    for k in range(0, 512):
        # q_val = q_ptr[k]
        q_val = tl.load(q_ptr + k)
        # k_vec = K_ptr[pid_m, k] = *(K_ptr + pid_m * 512 + k)
        k_val = tl.load(K_ptr + pid_m * 512 + k)
        acc += q_val * k_val
    # Store y[pid_m] = acc
    tl.store(y_ptr + pid_m, acc)


# GEMV kernel for D_in=64: y[M] = q_row[64] @ Kp_rows[M, 64].T
@triton.jit
def row_gemv_64_kernel(q_ptr,  # *float32, q_row of length 64
                       K_ptr,  # *float32, K_rows[M, 64] contiguous
                       y_ptr,  # *float32, output vector [M]
                       M: tl.constexpr):
    pid_m = tl.program_id(0)
    acc = tl.zeros([1], dtype=tl.float32)
    for k in range(0, 64):
        q_val = tl.load(q_ptr + k)
        k_val = tl.load(K_ptr + pid_m * 64 + k)
        acc += q_val * k_val
    tl.store(y_ptr + pid_m, acc)


# 1D scale: y[M] = x[M] * scale
@triton.jit
def scale_1d_kernel(x_ptr,  # *float32, input vector [M]
                    y_ptr,  # *float32, output vector [M]
                    scale: tl.float32,
                    M: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    y = x * scale
    tl.store(y_ptr + pid, y)


# 1D apply negative infinity where j <= pos: out[j] = -inf if j <= pos else x[j]
@triton.jit
def mask_neg_inf_1d_kernel(x_ptr,  # *float32, input vector [M]
                           y_ptr,  # *float32, output vector [M]
                           pos: tl.int32,
                           M: tl.constexpr):
    pid = tl.program_id(0)
    j = pid  # pid is the token index
    x = tl.load(x_ptr + j)
    cond = j <= pos
    y = tl.where(cond, -float('inf'), x)
    tl.store(y_ptr + j, y)


# 1D logsumexp: lse = log(sum(exp(x - max))) / ln(2)
@triton.jit
def lse_row_kernel(x_ptr,  # *float32, input vector [M]
                   lse_ptr,  # *float32, single scalar output
                   M: tl.constexpr):
    maxv = -float('inf')
    for j in range(0, M):
        v = tl.load(x_ptr + j)
        maxv = tl.maximum(maxv, v)
    sumexp = 0.0
    ln2 = 1.4426950408889634  # 1 / ln(2)
    for j in range(0, M):
        v = tl.load(x_ptr + j)
        sumexp += tl.exp(v - maxv)
    lse = tl.log(sumexp) * ln2
    tl.store(lse_ptr, lse)


# 1D softmax with mask (set -inf positions to 0 after shift)
@triton.jit
def softmax_masked_1d_kernel(x_ptr,  # *float32, input vector [M]
                             out_ptr,  # *float32, output vector [M]
                             pos: tl.int32,
                             M: tl.constexpr):
    maxv = -float('inf')
    for j in range(0, M):
        v = tl.load(x_ptr + j)
        maxv = tl.maximum(maxv, v)
    for j in range(0, M):
        v = tl.load(x_ptr + j)
        v = tl.where(j <= pos, -float('inf'), v - maxv)
        expv = tl.exp(v)
        tl.store(out_ptr + j, expv)


# GEMV kernel: out[Dn] = v[M] @ K_rows[M, Dn].T, here Dn=512
@triton.jit
def gemv_row_512_kernel(v_ptr,  # *float32, input vector [M]
                        K_ptr,  # *float32, K_rows[M, 512] contiguous
                        out_ptr,  # *float32, output vector [512]
                        M: tl.constexpr):
    pid_d = tl.program_id(0)  # each program handles one output feature along Dn
    acc = tl.zeros([1], dtype=tl.float32)
    for k in range(0, 512):
        v = tl.load(v_ptr + k)  # v[k]
        k_val = tl.load(K_ptr + k * M + pid_d)  # K[pid_d, k] but we iterate k over features
        # Correct indexing: K is [M, 512], row index is fixed by pid_d, iterate feature dim k
        # We need K_rows[pid_d, k] = *(K_ptr + pid_d * 512 + k)
        k_val = tl.load(K_ptr + pid_d * 512 + k)
        acc += v * k_val
    tl.store(out_ptr + pid_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
            and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA"

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # Constants
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute caches as float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Loop over batches and queries
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_len = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if q_start >= q_end or kv_len == 0:
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).tolist()
            M = len(tok_idx)

            # Gather rows of Kc and Kp for these tokens
            Kc_rows = Kc_all[tok_idx]  # [M, 512], float32
            Kp_rows = Kp_all[tok_idx]  # [M, 64], float32

            # Loop over queries in this batch
            for i in range(q_start, q_end):
                # Loop over heads
                for h in range(num_qo_heads):
                    # Load qn_row[h, :] and qp_row[h, :]
                    qn_vec = q_nope[i, h, :].to(torch.float32).contiguous()  # [512]
                    qp_vec = q_pe[i, h, :].to(torch.float32).contiguous()  # [64]

                    # 1) Compute qn_logits[M] via Triton GEMV
                    qn_logits = torch.empty(M, dtype=torch.float32, device=device)
                    row_gemv_512_kernel[(M,)](qn_vec, Kc_rows, qn_logits, M)

                    # 2) Compute qp_logits[M] via Triton GEMV
                    qp_logits = torch.empty(M, dtype=torch.float32, device=device)
                    row_gemv_64_kernel[(M,)](qp_vec, Kp_rows, qp_logits, M)

                    # 3) Sum logits
                    logits = torch.empty(M, dtype=torch.float32, device=device)
                    torch.add(qn_logits, qp_logits, out=logits)  # M must be constexpr for Triton; we use PyTorch for this add

                    # 4) Scale logits
                    logits_scaled = torch.empty(M, dtype=torch.float32, device=device)
                    scale_1d_kernel[(M,)](logits, logits_scaled, float(sm_scale), M)

                    # 5) Apply causal mask: set j <= (prefix_len + i) to -inf
                    prefix_len = kv_len - (q_end - q_start)
                    query_abs_pos = prefix_len + (i - q_start)  # absolute position for this query
                    logits_masked = torch.empty(M, dtype=torch.float32, device=device)
                    mask_neg_inf_1d_kernel[(M,)](logits_scaled, logits_masked, query_abs_pos, M)

                    # 6) Compute lse for this head
                    lse_val = torch.empty(1, dtype=torch.float32, device=device)
                    lse_row_kernel[(1,)](logits_masked, lse_val, M)
                    lse[i, h] = lse_val[0]

                    # 7) Compute softmax over masked logits (masked positions will be -inf -> exp(-inf)=0)
                    softmax_out = torch.empty(M, dtype=torch.float32, device=device)
                    softmax_masked_1d_kernel[(M,)](logits_masked, softmax_out, query_abs_pos, M)

                    # 8) Compute out[h, :] = softmax_out @ Kc_rows.T
                    out_vec = torch.empty(512, dtype=torch.float32, device=device)
                    gemv_row_512_kernel[(512,)](softmax_out, Kc_rows, out_vec, M)

                    # 9) Store output as bfloat16
                    output[i, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
