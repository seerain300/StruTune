import torch
import math

import triton
import triton.language as tl


# 1) GEMV kernel: compute y[M] = q_row @ K[M, D_in].T
# Each program handles one output element along M; we loop over D_in to accumulate.
# Note: Triton kernels must be defined at class scope to be visible to forward.
@triton.jit
def row_gemv_kernel(q_ptr,         # *float32, row vector of length D_in (1D)
                    K_ptr,         # *float32, matrix [M, D_in] flattened
                    y_ptr,         # *float32, output vector [M]
                    M: tl.constexpr,      # number of rows in K
                    D_in: tl.constexpr):  # length of the row vector
    pid_m = tl.program_id(0)  # each program handles one token index
    acc = tl.zeros([1], dtype=tl.float32)
    # Loop over D_in and accumulate dot product
    for k in range(0, D_in):
        qk = tl.load(q_ptr + k)  # load scalar q[k]
        # For each token, compute K[pid_m, k] and accumulate
        K_elem = tl.load(K_ptr + pid_m * D_in + k)  # K_ptr is row-major [M, D_in]
        acc += qk * K_elem
    tl.store(y_ptr + pid_m, acc)


# 2) Scale 1D vector by scalar
@triton.jit
def scale_1d_kernel(x_ptr,  # *float32
                    y_ptr,  # *float32
                    scale: tl.float32,
                    N: tl.constexpr):
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        yi = xi * scale
        tl.store(y_ptr + i, yi)


# 3) Apply causal mask: y[i] = x[i] if i > pos else -inf (elementwise)
@triton.jit
def mask_neg_inf_1d_kernel(x_ptr,  # *float32
                            y_ptr,  # *float32
                            N: tl.constexpr,
                            pos: tl.int32):
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        keep = i > pos
        yi = tl.where(keep, xi, -float("inf"))
        tl.store(y_ptr + i, yi)


# 4) LSE of masked 1D vector: lse = log(sum(exp(x))) / ln(2)
@triton.jit
def lse_row_kernel(x_ptr,  # *float32, masked logits
                    out_ptr,  # *float32, scalar output
                    N: tl.constexpr):
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        max_val = tl.maximum(max_val, xi)
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        sum_exp += tl.exp(xi - max_val)
    ln2 = 0.6931471805599453  # log(2)
    lse = tl.log(sum_exp) / ln2 + max_val
    tl.store(out_ptr, lse)


# 5) Softmax over masked 1D vector (elementwise, with causal mask: keep i > pos)
@triton.jit
def softmax_masked_row_kernel(x_ptr,  # *float32
                               out_ptr,  # *float32
                               N: tl.constexpr,
                               pos: tl.int32):
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        max_val = tl.maximum(max_val, xi)
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        keep = i > pos
        sum_exp += tl.where(keep, tl.exp(xi - max_val), 0.0)
    for i in range(0, N):
        xi = tl.load(x_ptr + i)
        keep = i > pos
        numerator = tl.where(keep, tl.exp(xi - max_val), 0.0)
        out_i = numerator / sum_exp
        tl.store(out_ptr + i, out_i)


# 6) GEMV of 1D vector with 2D matrix (columns): out[D_in] = softmax[M] @ Kc[M, D_in].T
#    Each program handles one output feature (along D_in)
@triton.jit
def gemv_row_kernel(softmax_ptr,   # *float32, 1D vector [M]
                    K_ptr,         # *float32, matrix [M, D_in] flattened
                    out_ptr,       # *float32, output vector [D_in]
                    M: tl.constexpr,
                    D_in: tl.constexpr):
    pid_d = tl.program_id(0)  # each program handles one feature k in D_in
    acc = tl.zeros([1], dtype=tl.float32)
    for m in range(0, M):
        sm = tl.load(softmax_ptr + m)
        K_elem = tl.load(K_ptr + m * D_in + pid_d)
        acc += sm * K_elem
    tl.store(out_ptr + pid_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # All computation must be done via Triton kernels in forward; no torch ops.
        device = q_nope.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA tensors"

        # Dimensions (constants per problem setup)
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        # Total queries and batch sizes
        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Prepare output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            if q_len <= 0 or kv_len <= 0:
                continue

            # Gather Kc and Kp for this batch using indices
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)
            Kc = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [kv_len, 512]
            Kp = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [kv_len, 64]

            # prefix_len: number of previously cached tokens (tokens before current query)
            prefix_len = kv_len - q_len

            # Process each query i in this batch
            for i in range(q_len):
                # Loop over heads h
                for h in range(num_qo_heads):
                    # Load qn and qp row vectors (float32 for kernels)
                    qn_vec = q_nope[q_start + i, h, :].to(torch.float32).contiguous()  # [512]
                    qp_vec = q_pe[q_start + i, h, :].to(torch.float32).contiguous()   # [64]

                    # 1) Compute qn_logits[M] = qn_vec @ Kc.T via Triton GEMV
                    qn_logits = torch.empty(kv_len, dtype=torch.float32, device=device)
                    row_gemv_kernel[(kv_len,)](qn_vec, Kc, qn_logits, kv_len, head_dim_ckv)

                    # 2) Compute qp_logits[M] = qp_vec @ Kp.T via Triton GEMV
                    qp_logits = torch.empty(kv_len, dtype=torch.float32, device=device)
                    row_gemv_kernel[(kv_len,)](qp_vec, Kp, qp_logits, kv_len, head_dim_kpe)

                    logits = qn_logits + qp_logits  # [kv_len], float32

                    # 3) Scale logits by sm_scale
                    logits_scaled = torch.empty(kv_len, dtype=torch.float32, device=device)
                    scale_1d_kernel[(kv_len,)](logits, logits_scaled, sm_scale, kv_len)

                    # 4) Apply causal mask: j > (prefix_len + i)
                    query_abs_pos = prefix_len + i
                    logits_masked = torch.empty(kv_len, dtype=torch.float32, device=device)
                    mask_neg_inf_1d_kernel[(kv_len,)](logits_scaled, logits_masked, kv_len, query_abs_pos)

                    # 5) Compute lse per head (base 2): lse[h] = logsumexp(logits_masked)
                    lse_val = torch.empty(1, dtype=torch.float32, device=device)
                    lse_row_kernel[(1,)](logits_masked, lse_val, kv_len)
                    lse[q_start + i, h] = lse_val[0]

                    # 6) Softmax over masked logits
                    softmax_out = torch.empty(kv_len, dtype=torch.float32, device=device)
                    softmax_masked_row_kernel[(kv_len,)](logits_masked, softmax_out, kv_len, query_abs_pos)

                    # 7) Final GEMV: out[h, :] = softmax_out @ Kc
                    out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                    gemv_row_kernel[(head_dim_ckv,)](softmax_out, Kc, out_row, kv_len, head_dim_ckv)

                    # Store output as bfloat16
                    output[q_start + i, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
