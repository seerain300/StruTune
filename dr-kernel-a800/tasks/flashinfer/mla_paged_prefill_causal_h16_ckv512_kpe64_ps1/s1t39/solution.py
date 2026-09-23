import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# A: [M, K], B: [K, N], C: [M, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Load A and B tiles, using masks to avoid OOB if partial tiles
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)  # BLOCK_M x BLOCK_K x BLOCK_K x BLOCK_N

    # Write back to C
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Row-wise softmax with causal mask: positions j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_outn: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK)
    x_ptrs = X_ptr + row_id * stride_xn + j_offsets
    out_ptrs = Out_ptr + row_id * stride_outn + j_offsets
    mask = j_offsets < N

    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    causal = j_offsets > absolute_pos
    x = tl.where(causal & mask, -float("inf"), x)

    m = tl.max(x, axis=0)
    x = x - m
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp

    tl.store(out_ptrs, out, mask=mask)


# Row-wise logsumexp (base-2) with causal mask
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_outn: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK)
    x_ptrs = X_ptr + row_id * stride_xn + j_offsets
    out_ptrs = Out_ptr + row_id * stride_outn

    mask = j_offsets < N
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    causal = j_offsets > absolute_pos
    x = tl.where(causal & mask, -float("inf"), x)

    m = tl.max(x, axis=0)
    # sum exp(x - m) over valid positions
    sum_exp = tl.sum(tl.exp(x - m), axis=0)
    lse = tl.log(sum_exp) / tl.log(2.0)
    tl.store(out_ptrs, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        total_q = q_nope.shape[0]
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        num_qo_heads = 16
        head_dim_ckv = q_nope.shape[2]
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        head_dim_kpe = q_pe.shape[2]
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        num_kv_indices = kv_indices.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Prepare Kc_all and Kp_all: [num_pages, dim]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        output = torch.zeros(
            (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg

            if q_len <= 0 or kv_len <= 0:
                continue

            # Collect token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query in this batch
            for i in range(q_len):
                abs_q = q_start + i
                # qn: [16, 512], qp: [16, 64]
                qn = q_nope[abs_q].to(torch.float32).contiguous()  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32).contiguous()   # [16, 64]

                # scores_n = qn @ Kc.T -> [16, kv_len]
                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_n = torch.matmul(qn, Kc.transpose(0, 1))  # torch op for matmul
                scores_p = torch.matmul(qp, Kp.transpose(0, 1))  # torch op for matmul
                scores = scores_n + scores_p  # [16, kv_len]
                scores = scores.to(torch.float32)

                prefix_len = kv_len - q_len  # number of previously processed tokens
                absolute_pos = prefix_len + i  # absolute position for this query in the sequence

                # Softmax with causal mask (row-wise)
                # Ensure BLOCK covers N
                BLOCK = 1 << (kv_len - 1).bit_length()  # next power of 2 >= kv_len
                softmax_out = torch.empty((16,), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, softmax_out,
                    kv_len, absolute_pos,
                    scores.stride(1), softmax_out.stride(0),
                    BLOCK=BLOCK
                )

                # Compute attention: attn = softmax(scores)
                attn = softmax_out  # [16] per head
                attn = attn.view(16, 1)  # [16, 1] for now

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                # Launch Triton matmul for a single row
                grid_mm = (triton.cdiv(16, 16), triton.cdiv(512, 64))
                matmul_kernel[grid_mm](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64
                )

                output[abs_q] = out_row.to(torch.bfloat16)

                # LSE base-2 with causal mask: lse per head row
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, absolute_pos,
                    scores.stride(1), lse_row.stride(0),
                    BLOCK=BLOCK
                )
                lse[abs_q] = lse_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
