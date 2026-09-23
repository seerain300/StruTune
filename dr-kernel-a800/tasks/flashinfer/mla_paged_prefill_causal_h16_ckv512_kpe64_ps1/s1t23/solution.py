import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
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

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)
        # Load B tile: [BLOCK_K, BLOCK_N], B has shape (K, N)
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Softmax per row with causal mask: j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,           # typically 1.0
    absolute_pos: tl.int32,      # positions j > absolute_pos should be masked as -inf
    BLOCK: tl.constexpr,         # must be power-of-two (e.g., 128)
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    # Load row
    x = tl.load(X_ptr + row_id * N + cols, mask=cols < N, other=0.0)
    # Apply causal mask: j > absolute_pos -> -inf
    j = cols
    mask_true = j <= absolute_pos
    x = tl.where(mask_true, x, -float("inf"))
    # Stable softmax
    x = x * scale
    m = tl.max(x, axis=0)
    x = x - m
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    # Store
    tl.store(Out_ptr + row_id * N + cols, out, mask=cols < N)


# Logsumexp per row with causal mask (base-2): out = log(sum(exp(scores))) / ln(2)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,           # typically 1.0
    absolute_pos: tl.int32,      # positions j > absolute_pos should be masked as -inf
    BLOCK: tl.constexpr,         # must be power-of-two (e.g., 128)
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + cols, mask=cols < N, other=0.0)
    j = cols
    mask_true = j <= absolute_pos
    x = tl.where(mask_true, x, -float("inf"))
    x = x * scale
    m = tl.max(x, axis=0)
    x = x - m
    sum_exp = tl.sum(tl.exp(x), axis=0)
    out = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(Out_ptr + row_id, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors on CUDA device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device"
        device = q_nope.device

        # Shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages = ckv_cache.shape[0]
        # Constants
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        # Build Kc_all, Kp_all from caches (only need to read, no computation)
        Kc_all = ckv_cache.to(torch.float32)          # [num_pages, 1, 512] -> [num_pages, 512]
        Kp_all = kpe_cache.to(torch.float32)          # [num_pages, 1, 64]  -> [num_pages, 64]

        # Prepare output buffers (fp32 for compute, bf16 for return)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue
            kv_len = page_end - page_beg

            # Gather tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # [kv_len]
            Kc = Kc_all[tok_idx].to(torch.float32)                   # [kv_len, 512]
            Kp = Kp_all[tok_idx].to(torch.float32)                   # [kv_len, 64]

            # Process each query in the batch
            for i in range(q_len):
                abs_q = q_start + i
                # Load q vectors
                qn = q_nope[abs_q].to(torch.float32)                # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)                  # [16, 64]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                # We need B with shape (K, N) i.e., Kc.T
                KcT = Kc.transpose(0, 1).contiguous()               # [512, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid1 = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid1](
                    qn, KcT, scores_n,
                    16, kv_len, 512,
                    qn.stride(0), qn.stride(1),
                    KcT.stride(0), KcT.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                KpT = Kp.transpose(0, 1).contiguous()               # [64, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid2 = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid2](
                    qp, KpT, scores_p,
                    16, kv_len, 64,
                    qp.stride(0), qp.stride(1),
                    KpT.stride(0), KpT.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p                        # [16, kv_len]

                # Apply causal mask: positions j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i
                # Compute softmax and lse
                attn_scores = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(1,)](
                    scores, attn_scores,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=128
                )

                # lse per head (base-2)
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(1,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=128
                )
                lse[abs_q] = lse_row

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                KcT2 = Kc.transpose(0, 1).contiguous()  # [512, kv_len]
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(512, 64),)](
                    attn_scores, KcT2, out_row,
                    16, kv_len, 512,
                    attn_scores.stride(0), attn_scores.stride(1),
                    KcT2.stride(0), KcT2.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

        # Cast output to bfloat16 as requested by original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
