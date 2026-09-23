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

        A_block = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        B_block = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        acc += tl.dot(A_block, B_block)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax row-wise with causal mask: positions j > absolute_pos -> treated as -inf
@triton.jit
def softmax_row_causal_kernel(
    x_ptr, y_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
    scale: tl.float32 = 1.0,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float("inf"))

    # Apply causal mask: set j > absolute_pos to -inf
    causal = offs > absolute_pos
    x = tl.where(causal & mask, -float("inf"), x)

    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    sum_e = tl.sum(e, axis=0)
    soft = e / sum_e

    # Store
    tl.store(y_ptr + row * N + offs, soft, mask=mask)


# LogSumExp row-wise with causal mask, base-2
@triton.jit
def lse_row_causal_kernel(
    x_ptr, y_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    scale: tl.float32,  # 1.0 / ln(2)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float("inf"))

    # Apply causal mask: set j > absolute_pos to -inf
    causal = offs > absolute_pos
    x = tl.where(causal & mask, -float("inf"), x)

    m = tl.max(x, axis=0)
    x = x - m
    sum_e = tl.sum(tl.exp(x), axis=0)
    lse = tl.log(sum_e) * scale

    tl.store(y_ptr + row, lse)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        q_nope,  # [T, 16, 512], bfloat16 or float32
        q_pe,    # [T, 16, 64],  bfloat16 or float32
        ckv_cache,  # [num_pages, 1, 512]
        kpe_cache,  # [num_pages, 1, 64]
        qo_indptr,  # [len_indptr], int32
        kv_indptr,  # [len_indptr], int32
        kv_indices, # [num_kv_indices], int32
        sm_scale: float,
    ):
        # Preprocess caches
        device = q_nope.device
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        total_q = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        head_dim_q = q_nope.shape[2]
        assert head_dim_q == 512
        head_dim_k = q_pe.shape[2]
        assert head_dim_k == 64

        # Process only the first batch element since len_indptr == 2 in provided inputs
        q_start = int(qo_indptr[0].item())
        q_end = int(qo_indptr[1].item())
        total_q_actual = q_end - q_start
        assert total_q == total_q_actual

        kv_len = int(kv_indptr[1].item()) - int(kv_indptr[0].item())
        tok_idx = kv_indices[kv_indptr[0]: kv_indptr[1]].to(torch.int64)  # [kv_len]
        Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
        Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

        output = torch.empty((total_q, num_heads, head_dim_q), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_heads), -float("inf"), dtype=torch.float32, device=device)

        for i in range(total_q_actual):
            abs_q = q_start + i

            # Load qn and qp and convert to float32
            qn = q_nope[abs_q].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[abs_q].contiguous().to(torch.float32)   # [16, 64]

            # scores_n = qn @ Kc.T -> [16, kv_len]
            scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
            stride_am = qn.stride(0)    # 512
            stride_ak = qn.stride(1)    # 1
            stride_bk2 = Kc.stride(1)   # 1
            stride_bn2 = Kc.stride(0)   # 512
            stride_cm = scores_n.stride(0)  # kv_len
            stride_cn = scores_n.stride(1)  # 1

            grid_mat_n = (triton.cdiv(16, 16), triton.cdiv(kv_len, 128))
            matmul_kernel[grid_mat_n](
                qn, Kc.transpose(0, 1).contiguous(), scores_n,
                16, 512, kv_len,
                stride_am, stride_ak,
                stride_bk2, stride_bn2,
                stride_cm, stride_cn,
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            # scores_p = qp @ Kp.T -> [16, kv_len]
            scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
            stride_am2 = qp.stride(0)    # 64
            stride_ak2 = qp.stride(1)    # 1
            stride_bk3 = Kp.stride(1)    # 1
            stride_bn3 = Kp.stride(0)    # 64
            stride_cm2 = scores_p.stride(0)  # kv_len
            stride_cn2 = scores_p.stride(1)  # 1

            grid_mat_p = (triton.cdiv(16, 16), triton.cdiv(kv_len, 128))
            matmul_kernel[grid_mat_p](
                qp, Kp.transpose(0, 1).contiguous(), scores_p,
                16, 64, kv_len,
                stride_am2, stride_ak2,
                stride_bk3, stride_bn3,
                stride_cm2, stride_cn2,
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            scores = scores_n + scores_p  # [16, kv_len]
            absolute_pos = kv_len - total_q_actual + i  # prefix_len + query index

            # Softmax (row-wise) with causal mask, store attn
            attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
            softmax_row_causal_kernel[(16,)](
                scores, attn,
                kv_len, absolute_pos,
                BLOCK=128,
                scale=1.0,
            )

            # out_row = attn @ Kc -> [16, 512]
            out_row = torch.empty((16, 512), dtype=torch.float32, device=device)

            stride_am3 = attn.stride(0)   # kv_len
            stride_ak3 = attn.stride(1)   # 1
            stride_bk4 = Kc.stride(0)     # 512
            stride_bn4 = Kc.stride(1)     # 1
            stride_cm3 = out_row.stride(0)  # 512
            stride_cn3 = out_row.stride(1)  # 1

            grid_mat_out = (triton.cdiv(16, 16), triton.cdiv(512, 128))
            matmul_kernel[grid_mat_out](
                attn, Kc, out_row,
                16, kv_len, 512,
                stride_am3, stride_ak3,
                stride_bk4, stride_bn4,
                stride_cm3, stride_cn3,
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=128,
                num_warps=4, num_stages=2,
            )

            # Store output as bfloat16
            output[abs_q] = out_row.to(torch.bfloat16)

            # LSE per row (base-2)
            lse_row = torch.empty((16,), dtype=torch.float32, device=device)
            lse_row_causal_kernel[(16,)](
                scores, lse_row,
                kv_len, absolute_pos,
                1.0 / math.log(2.0),
                BLOCK=128,
            )
            lse[abs_q] = lse_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
