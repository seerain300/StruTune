import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A: [M, K], B: [K, N]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        a = tl.load(A_ptrs, mask=A_mask, other=0.0)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store result
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Softmax with causal mask per row: Out = softmax(scale * X) with j > absolute_pos => -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,            # typically 1.0
    absolute_pos: tl.int32,       # if j > absolute_pos, set to -inf
    BLOCK: tl.constexpr,          # power-of-two, e.g., 1024
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load row
    x = tl.load(X_ptr + row_id * N + cols, mask=mask, other=0.0)

    # Apply causal mask: j > absolute_pos -> -inf
    causal = cols > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    # Scale
    x = x * scale

    # Stable softmax: subtract max
    row_max = tl.max(x, axis=0)
    x = x - row_max

    # Exponentiate
    e = tl.exp(x)

    # Sum
    denom = tl.sum(e, axis=0)

    # Normalize
    out = e / denom

    # Store per row vector
    tl.store(Out_ptr + row_id * N + cols, out, mask=mask)


# logsumexp base-2 with causal mask per row: Out[row] = log(sum(exp(scale * X))) / ln(2), j > absolute_pos -> -inf
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,            # typically 1.0
    absolute_pos: tl.int32,       # if j > absolute_pos, set to -inf
    BLOCK: tl.constexpr,          # power-of-two, e.g., 1024
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row_id * N + cols, mask=mask, other=0.0)

    # Apply causal mask
    causal = cols > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    # Scale and stable logsumexp in natural log
    x_scaled = x * scale
    row_max = tl.max(x_scaled, axis=0)
    x_shifted = x_scaled - row_max
    e = tl.exp(x_shifted)
    sum_e = tl.sum(e, axis=0)
    lse_ln = tl.log(sum_e)
    # Convert to base-2 log
    lse_base2 = lse_ln / tl.log(2.0)

    # Store scalar per row
    tl.store(Out_ptr + row_id, lse_base2)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be CUDA tensors"
        device = q_nope.device

        # Convert caches to fp32 contiguous
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, head_dim_kpe]

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64

        # Precompute output and lse buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch in qo_indptr
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            if q_start >= q_end:
                continue

            # KV indices and keys for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # int64 for safe indexing
            kv_len = tok_idx.shape[0]

            Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

            # Extract q_batch
            qn_batch = q_nope[q_start:q_end]  # [q_len, 16, 512]
            qp_batch = q_pe[q_start:q_end]    # [q_len, 16, 64]
            q_len = qn_batch.shape[0]

            for i in range(q_len):
                abs_q = q_start + i
                # Matmul qn @ Kc.T -> [16, kv_len]
                qn = qn_batch[i].contiguous().to(torch.float32)  # [16, 512]
                KcT = Kc.transpose(0, 1).contiguous()  # [512, kv_len]
                attn_scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # Launch matmul kernel
                M = 16; K = 512; N = kv_len
                grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
                matmul_kernel[grid](
                    qn, KcT, attn_scores_n,
                    M, K, N,
                    qn.stride(0), qn.stride(1),
                    KcT.stride(0), KcT.stride(1),
                    attn_scores_n.stride(0), attn_scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64
                )

                # Matmul qp @ Kp.T -> [16, kv_len]
                qp = qp_batch[i].contiguous().to(torch.float32)  # [16, 64]
                KpT = Kp.transpose(0, 1).contiguous()  # [64, kv_len]
                attn_scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid2 = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid2](
                    qp, KpT, attn_scores_p,
                    16, 64, kv_len,
                    qp.stride(0), qp.stride(1),
                    KpT.stride(0), KpT.stride(1),
                    attn_scores_p.stride(0), attn_scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64
                )

                attn_scores = attn_scores_n + attn_scores_p  # [16, kv_len]

                # Compute prefix_len and absolute_pos for causal mask
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Softmax with causal mask
                out_attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_softmax = (1,)
                softmax_row_causal_kernel[grid_softmax](
                    attn_scores, out_attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=1024
                )

                # Out = attn @ Kc
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                KcT2 = Kc.transpose(0, 1).contiguous()  # [512, kv_len]
                grid3 = (triton.cdiv(16, 16), triton.cdiv(512, 64))
                matmul_kernel[grid3](
                    out_attn, KcT2, out_row,
                    16, kv_len, 512,
                    out_attn.stride(0), out_attn.stride(1),
                    KcT2.stride(0), KcT2.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64
                )

                # Store output for this query
                output[abs_q] = out_row  # fp32 buffer; cast later

                # LSE per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(1,)](
                    attn_scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=1024
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as requested by original
        output = output.to(torch.bfloat16)
        return output, lse


# Optional: original Model can be alias to ModelNew
class Model(ModelNew):
    pass


def run(*args):
    return ModelNew()(*args)
