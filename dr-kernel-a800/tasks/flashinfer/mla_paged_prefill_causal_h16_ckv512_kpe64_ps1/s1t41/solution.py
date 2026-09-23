import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# Positional: A_ptr, B_ptr, C_ptr, M, N, K
# Keyword: stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
# Compile-time constants: BLOCK_M, BLOCK_N, BLOCK_K
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
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

        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak   # [BM, BK]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn   # [BK, BN]

        A = tl.load(A_ptrs, mask=None, other=0.0)
        B = tl.load(B_ptrs, mask=None, other=0.0)

        acc += tl.dot(A, B)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=True)


# Softmax row-wise with causal mask: j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    in_ptr, out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    scale: tl.float32,  # not used here
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + row_id * N + offsets, mask=mask, other=0.0)
    # causal mask: positions j > absolute_pos => -inf
    causal = offsets > absolute_pos
    x = tl.where(causal, -float("inf"), x)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    soft = num * (1.0 / den)
    tl.store(out_ptr + row_id * N + offsets, soft, mask=mask)


# LogSumExp row-wise with causal mask: base-2 LSE
@triton.jit
def lse_row_causal_kernel(
    in_ptr, out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    scale: tl.float32,  # not used here
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + row_id * N + offsets, mask=mask, other=0.0)
    causal = offsets > absolute_pos
    x = tl.where(causal, -float("inf"), x)
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    exp_x = tl.exp(x_shift)
    sum_exp = tl.sum(exp_x, axis=0)
    lse = tl.log(sum_exp) * (1.0 / math.log(2.0))
    tl.store(out_ptr + row_id, lse)  # one scalar per row


# Matmul kernel specialized for row-wise product: output[M, N] = A[M, K] @ B[K, N]
# Here A is row-sized and B is [K, N], e.g., M=1 (per row).
@triton.jit
def matmul_row_NK_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
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

        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak   # [BM, BK]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn   # [BK, BN]

        A = tl.load(A_ptrs, mask=None, other=0.0)
        B = tl.load(B_ptrs, mask=None, other=0.0)

        acc += tl.dot(A, B)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be on CUDA device"
        device = q_nope.device

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # Constants (as in original code)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Allocate outputs (fp32 for compute, bf16 for output)
        output = torch.empty((total_q, 16, 512), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg

            if q_len <= 0 or kv_len <= 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]
            KcT = Kc.transpose(0, 1).contiguous()  # [512, kv_len]
            KpT = Kp.transpose(0, 1).contiguous()  # [64, kv_len]

            for i in range(q_len):
                abs_q = q_start + i

                # q_nope[i] and q_pe[i]: [16, 512] and [16, 64]
                qn = q_nope[abs_q].to(torch.float32).contiguous()  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32).contiguous()    # [16, 64]

                # scores_n = qn @ KcT  -> [16, kv_len]
                M = qn.shape[0]          # 16
                N = KcT.shape[1]         # kv_len
                K = qn.shape[1]          # 512

                out_scores_n = torch.empty((M, N), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(M, 16), triton.cdiv(N, 64))
                matmul_kernel[grid_n](
                    qn, KcT, out_scores_n,
                    M, N, K,
                    qn.stride(0), qn.stride(1),
                    KcT.stride(0), KcT.stride(1),
                    out_scores_n.stride(0), out_scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2,
                )

                # scores_p = qp @ KpT -> [16, kv_len]
                out_scores_p = torch.empty((M, N), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(M, 16), triton.cdiv(N, 64))
                matmul_kernel[grid_p](
                    qp, KpT, out_scores_p,
                    M, N, KpT.shape[1],
                    qp.stride(0), qp.stride(1),
                    KpT.stride(0), KpT.stride(1),
                    out_scores_p.stride(0), out_scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2,
                )

                scores = out_scores_n + out_scores_p  # [16, kv_len]

                # Causal mask: positions j > (prefix_len + i) => -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Softmax with causal mask (row-wise), output is fp32
                attn = torch.empty((M, N), dtype=torch.float32, device=device)
                grid_s = (M,)
                softmax_row_causal_kernel[grid_s](
                    scores, attn,
                    N, absolute_pos,
                    1.0,  # scale not used
                    BLOCK=128,  # power-of-two
                    num_warps=1,
                )

                # LSE per head (base-2), one scalar per head
                lse_row = torch.empty((M,), dtype=torch.float32, device=device)
                grid_l = (M,)
                lse_row_causal_kernel[grid_l](
                    scores, lse_row,
                    N, absolute_pos,
                    1.0,  # scale not used
                    BLOCK=128,
                    num_warps=1,
                )
                lse[abs_q] = lse_row  # [16]

                # Output: attn @ Kc -> [16, 512], then cast to bfloat16
                M_out = M
                N_out = 512
                K


def run(*args):
    return ModelNew()(*args)
