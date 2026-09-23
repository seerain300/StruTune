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
    # 2D tiling
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


# Softmax per row with causal mask: positions j > absolute_pos set to -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # typically 1.0
    absolute_pos: tl.int32,     # starting from 0
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid_mask = cols < N
    x_ptrs = X_ptr + row_id * N + cols
    out_ptrs = Out_ptr + row_id * N + cols

    # Load row, invalid positions get 0 (we'll ignore via mask)
    x = tl.load(x_ptrs, mask=valid_mask, other=0.0)
    # Apply causal mask: j > absolute_pos -> -inf
    causal_mask = cols > absolute_pos
    x_masked = tl.where(causal_mask, -float("inf"), x)

    # Stable softmax
    x_max = tl.max(x_masked, axis=0)
    x_shifted = x_masked - x_max
    exp_x = tl.exp(x_shifted)
    exp_x = tl.where(valid_mask, exp_x, 0.0)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp

    tl.store(out_ptrs, out, mask=valid_mask)


# Logsumexp per row with causal mask, base-2: Out = log(sum(exp(x))) / ln(2)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # not used here
    absolute_pos: tl.int32,     # starting from 0
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid_mask = cols < N
    x_ptrs = X_ptr + row_id * N + cols
    out_ptrs = Out_ptr + row_id

    # Load row
    x = tl.load(x_ptrs, mask=valid_mask, other=0.0)
    # Apply causal mask: j > absolute_pos -> -inf
    causal_mask = cols > absolute_pos
    x_masked = tl.where(causal_mask, -float("inf"), x)

    # Stable logsumexp
    x_max = tl.max(x_masked, axis=0)
    sum_exp = tl.sum(tl.exp(x_masked - x_max), axis=0)
    lse = tl.log(sum_exp) / tl.log(2.0)

    tl.store(out_ptrs, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move tensors to CUDA and ensure float32 for computation
        device = q_nope.device
        assert device.type == 'cuda', "ModelNew expects CUDA tensors"
        q_nope = q_nope.to(torch.float32)
        q_pe = q_pe.to(torch.float32)
        ckv_cache = ckv_cache.to(torch.float32)
        kpe_cache = kpe_cache.to(torch.float32)

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate over batches
        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len == 0:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg

            if kv_len == 0:
                # No kv tokens for this batch, fill defaults and skip
                for i in range(q_len):
                    output[q_start + i].zero_()
                    lse[q_start + i].zero_()
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices into Kc_all/Kp_all
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query in the batch
            for i in range(q_len):
                abs_q = q_start + i
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]

                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid1 = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid1](
                    qn, Kc.transpose(0, 1).contiguous(), scores_n,
                    16, kv_len, 512,
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid2 = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid2](
                    qp, Kp.transpose(0, 1).contiguous(), scores_p,
                    16, kv_len, 64,
                    qp.stride(0), qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # Causal mask: j > (prefix_len + i) -> -inf, prefix_len = kv_len - q_len
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # Apply softmax with causal mask
                softmax_row_causal_kernel[(1,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=128  # power-of-two
                )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                grid3 = (triton.cdiv(16, 16), triton.cdiv(512, 64))
                matmul_kernel[grid3](
                    attn, Kc, out_row,
                    16, 512, kv_len,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # lse per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(1,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=128  # power-of-two
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as requested
        return output.to(torch.bfloat16), lse


# For environments expecting 'Model' class
class Model(ModelNew):
    pass


def run(*args):
    return ModelNew()(*args)
