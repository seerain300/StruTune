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
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Softmax with causal mask: row-wise, j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_outn: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_ptrs = X_ptr + row_id * stride_xn + cols * 0  # row base + cols
    out_ptrs = Out_ptr + row_id * stride_outn + cols * 0

    # Apply causal mask: positions j > (absolute_pos) -> -inf
    # j corresponds to cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    causal = cols > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(out_ptrs, out, mask=mask)


# LogSumExp base-2 with causal mask: row-wise
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_outn: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_ptrs = X_ptr + row_id * stride_xn + cols * 0
    out_ptrs = Out_ptr + row_id * stride_outn + cols * 0

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    causal = cols > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    m = tl.max(x, axis=0)
    x = x - m
    sum_e = tl.sum(tl.exp(x), axis=0)
    lse = tl.log(sum_e) / tl.log(2.0)
    tl.store(out_ptrs, lse, mask=True)  # scalar store, single element per row


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Expect CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        # Prepare Kc_all and Kp_all (already 1D per page)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers in fp32
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_size = qo_indptr.shape[0] - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len == 0:
                continue

            # KV indices and length
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Loop over queries in this batch
            for i in range(q_len):
                abs_q = q_start + i

                # qn: [16, 512], qp: [16, 64]
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)   # [16, 64]
                qn = qn.contiguous()
                qp = qp.contiguous()
                Kc = Kc.contiguous()
                Kp = Kp.contiguous()

                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_mm_n = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_mm_n](
                    qn, Kc.transpose(0, 1), scores_n,
                    num_qo_heads, kv_len, head_dim_ckv,
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_mm_p = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_mm_p](
                    qp, Kp.transpose(0, 1), scores_p,
                    num_qo_heads, kv_len, head_dim_kpe,
                    qp.stride(0), qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # scores = scores_n + scores_p
                scores = scores_n + scores_p  # [16, kv_len]
                scores = scores.contiguous()

                # Absolute position for causal mask: prefix_len + i
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Apply softmax with causal mask (row-wise)
                attn = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(num_qo_heads,)](
                    scores, attn,
                    kv_len, absolute_pos,
                    scores.stride(1), attn.stride(1),
                    BLOCK=128
                )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                grid_mm_out = (triton.cdiv(num_qo_heads, 16), triton.cdiv(head_dim_ckv, 64))
                matmul_kernel[grid_mm_out](
                    attn, Kc, out_row,
                    num_qo_heads, kv_len, head_dim_ckv,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # LSE base-2 with causal mask
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    kv_len, absolute_pos,
                    scores.stride(1), lse_row.stride(0),
                    BLOCK=128
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as requested
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
