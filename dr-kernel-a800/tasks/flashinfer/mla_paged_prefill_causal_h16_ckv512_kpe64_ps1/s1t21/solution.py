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

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)
        acc += tl.dot(a, b)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Softmax with causal mask per row: Out = softmax(X * scale) with j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # typically 1.0
    absolute_pos: tl.int32,     # j > absolute_pos => -inf
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load row
    x = tl.load(X_ptr + row_id * N + cols, mask=mask, other=-float("inf"))
    # Apply causal mask: j > absolute_pos -> -inf
    mask_causal = cols <= absolute_pos
    x = tl.where(mask_causal & mask, x, -float("inf"))

    # Stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = x * scale
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom
    tl.store(Out_ptr + row_id * N + cols, out, mask=mask)


# LogSumExp (base-2) with causal mask per row: Out = log(sum(exp(X - max))) / ln(2)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # not used here (logsumexp over original)
    absolute_pos: tl.int32,     # j > absolute_pos => -inf
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load row
    x = tl.load(X_ptr + row_id * N + cols, mask=mask, other=-float("inf"))
    # Apply causal mask: j > absolute_pos -> -inf
    mask_causal = cols <= absolute_pos
    x = tl.where(mask_causal & mask, x, -float("inf"))

    # Stable logsumexp
    x_max = tl.max(x, axis=0)
    x = x - x_max
    sum_exp = tl.sum(tl.exp(x), axis=0)
    # logsumexp in natural log, then divide by ln(2)
    lse_val = tl.log(sum_exp) / tl.log(2.0)
    tl.store(Out_ptr + row_id, lse_val)  # one value per row


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA device (environment should provide CUDA tensors)
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be CUDA tensors"

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Constants (same as original)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1 and ckv_cache.shape[2] == head_dim_ckv and kpe_cache.shape[2] == head_dim_kpe
        assert kv_indices.dim() == 1

        # Prepare Kc_all, Kp_all
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

        # Output and lse initialization
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_size = qo_indptr.shape[0] - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # KV range
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices into Kc_all/Kp_all
            Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
            Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

            # Q batch
            qn_batch = q_nope[q_start:q_end]  # [q_len, 16, 512]
            qp_batch = q_pe[q_start:q_end]    # [q_len, 16, 64]
            q_len = q_end - q_start

            for i in range(q_len):
                abs_q = q_start + i
                qn = qn_batch[i].to(torch.float32)  # [16, 512]
                qp = qp_batch[i].to(torch.float32) # [16, 64]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn, Kc, scores_n,
                    16, kv_len, 512,
                    qn.stride(0), qn.stride(1),   # A strides: (M=16,K=512) for qn, but we pass strides(0), strides(1)
                    Kc.stride(0), Kc.stride(1),   # B strides: (K=kv_len, N=512) for Kc.T -> Kc has (kv_len,512)
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_p](
                    qp, Kp, scores_p,
                    16, kv_len, 64,
                    qp.stride(0), qp.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Softmax with causal mask: apply -inf for j > absolute_pos
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(1,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos, BLOCK=kv_len
                )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                grid_mm = (triton.cdiv(16, 16), triton.cdiv(512, 64))
                matmul_kernel[grid_mm](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row  # keep as float32 for stability

                # lse per head: logsumexp(scores, dim=-1) / ln(2)
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos, BLOCK=kv_len
                )
                lse[abs_q] = lse_row

        # Return in the same dtypes as original outputs
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
