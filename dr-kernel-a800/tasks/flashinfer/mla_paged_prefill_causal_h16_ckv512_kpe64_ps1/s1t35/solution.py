import math
import torch
import triton
import triton.language as tl


# Tiled matmul kernel: C[M, N] = A[M, K] @ B[K, N]
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
    # Tile IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A and B tiles
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak  # [BM, BK]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn  # [BK, BN]

        # Load tiles with masks
        A_tile = tl.load(
            A_ptrs,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0
        )
        B_tile = tl.load(
            B_ptrs,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0
        )

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store results
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Row-wise softmax with causal mask along last dimension (columns)
# X: [1, N] pointer (we launch grid=(rows,), and X is a 1D vector by using row stride=0 for row)
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,  # absolute_pos: int32
    stride_xn: tl.int32, stride_outn: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x_ptrs = X_ptr + row * stride_xn + offs * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # causal mask: positions j > absolute_pos -> -inf
    causal_mask = offs > absolute_pos
    x = tl.where(causal_mask, -float('inf'), x)

    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    soft = exp_x / sum_exp

    out_ptrs = Out_ptr + row * stride_outn + offs * stride_outn
    tl.store(out_ptrs, soft, mask=mask)


# Row-wise logsumexp (base-2) with causal mask along last dimension (columns)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_out: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x_ptrs = X_ptr + row * stride_xn + offs * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    causal_mask = offs > absolute_pos
    x = tl.where(causal_mask, -float('inf'), x)

    m = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - m), axis=0)
    lse = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    # Store scalar per row
    out_ptr = Out_ptr + row * stride_out  # no need for column since scalar
    tl.store(out_ptr, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes
        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64
        device = q_nope.device

        # Convert caches to float32 and flatten batch dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_num = qo_indptr.shape[0] - 1  # len_indptr - 1 = number of batches
        for b in range(batch_num):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg

            if q_len == 0 or kv_len == 0:
                continue

            # Gather key vectors for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)
            Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

            # Prepare queries for this batch
            qn_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, 16, 512]
            qp_batch = q_pe[q_start:q_end].to(torch.float32)    # [q_len, 16, 64]

            prefix_len = kv_len - q_len  # number of cached tokens before this query

            for i in range(q_len):
                abs_q = q_start + i
                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn_batch[i], Kc.transpose(0, 1), scores_n,
                    num_qo_heads, kv_len, head_dim_ckv,
                    qn_batch[i].stride(0), qn_batch[i].stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_p](
                    qp_batch[i], Kp.transpose(0, 1), scores_p,
                    num_qo_heads, kv_len, head_dim_kpe,
                    qp_batch[i].stride(0), qp_batch[i].stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # Causal mask: j > (prefix_len + i) -> -inf
                absolute_pos = prefix_len + i
                attn = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(num_qo_heads,)](
                    scores, attn,
                    kv_len, absolute_pos,
                    scores.stride(1), attn.stride(1),
                    BLOCK=128
                )

                # out_row = attn @ Kc -> [16, 512]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                grid_mm = (triton.cdiv(num_qo_heads, 16), triton.cdiv(head_dim_ckv, 64))
                matmul_kernel[grid_mm](
                    attn, Kc, out_row,
                    num_qo_heads, kv_len, head_dim_ckv,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                output[abs_q] = out_row  # float32 buffer; we'll cast later

                # LogSumExp base-2 with causal mask
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    kv_len, absolute_pos,
                    scores.stride(1), lse_row.stride(0),
                    BLOCK=128
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


# The original get_inputs function can be reused; it provides typical shapes and indptrs.
# fused_operator wrapper for testing harness compatibility
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
