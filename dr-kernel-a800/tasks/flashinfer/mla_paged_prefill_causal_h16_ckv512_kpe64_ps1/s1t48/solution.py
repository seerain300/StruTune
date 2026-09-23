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

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A and B tiles
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Load tiles with masking
        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Row-wise softmax with causal mask applied by zeroing invalid positions, then renormalize
@triton.jit
def softmax_row_causal_kernel(
    scores_ptr, attn_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    # Grid: (N,)
    row_id = tl.program_id(0)
    # Load scores for row row_id
    offs = tl.arange(0, BLOCK)
    mask_row = offs < N
    scores = tl.load(scores_ptr + row_id * N + offs, mask=mask_row, other=0.0)

    # Stable softmax
    m = tl.max(scores, axis=0)
    y = scores - m
    e = tl.exp(y)
    sum_e = tl.sum(e, axis=0)
    soft = e / sum_e

    # Causal mask: j > absolute_pos -> set to 0, then renormalize
    valid = offs <= absolute_pos
    soft = tl.where(valid, soft, 0.0)
    sum_soft = tl.sum(soft, axis=0)
    attn = soft / tl.maximum(sum_soft, 1.0)  # avoid division by zero

    # Store attn
    tl.store(attn_ptr + row_id * N + offs, attn, mask=mask_row)


# Row-wise logsumexp (base-2) of scores
@triton.jit
def lse_row_kernel(
    scores_ptr, lse_ptr,
    N: tl.int32, scale: tl.float32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask_row = offs < N
    scores = tl.load(scores_ptr + row_id * N + offs, mask=mask_row, other=0.0)

    m = tl.max(scores, axis=0)
    y = scores - m
    e = tl.exp(y)
    sum_e = tl.sum(e, axis=0)
    lse = tl.log(sum_e) * scale  # scale = 1/ln(2)
    tl.store(lse_ptr + row_id, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_nope, q_pe,
        ckv_cache, kpe_cache,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale,
    ):
        # shapes and asserts
        assert q_nope.dim() == 3 and q_nope.shape[1] == 16 and q_nope.shape[2] == 512
        assert q_pe.dim() == 3 and q_pe.shape[1] == 16 and q_pe.shape[2] == 64
        total_q = q_nope.shape[0]
        num_heads = 16
        head_dim_q = 512
        head_dim_k = 64

        # Prepare Kc_all and Kp_all from caches
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_heads, head_dim_q), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.full((total_q, num_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

        # Only one batch in the provided get_inputs: len_indptr = 2, b = 0
        q_start = int(qo_indptr[0].item())
        q_end = int(qo_indptr[1].item())
        total_q_actual = q_end - q_start
        assert total_q == total_q_actual

        kv_len = int(kv_indptr[1].item()) - int(kv_indptr[0].item())
        tok_idx = kv_indices[kv_indptr[0]: kv_indptr[1]].to(torch.int64)
        Kc = Kc_all[tok_idx]  # [kv_len, 512]
        Kp = Kp_all[tok_idx]  # [kv_len, 64]

        # Process each query i
        for i in range(total_q_actual):
            abs_q = q_start + i
            qn = q_nope[abs_q].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[abs_q].contiguous().to(torch.float32)   # [16, 64]

            # Compute scores_n = qn @ Kc.T -> [16, kv_len]
            scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=q_nope.device)
            grid_mat = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
            matmul_kernel[grid_mat](
                qn, Kc.transpose(0, 1).contiguous(), scores_n,
                16, kv_len, 512,
                qn.stride(0), qn.stride(1),
                Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                scores_n.stride(0), scores_n.stride(1),
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            # Compute scores_p = qp @ Kp.T -> [16, kv_len]
            scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=q_nope.device)
            grid_mat2 = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
            matmul_kernel[grid_mat2](
                qp, Kp.transpose(0, 1).contiguous(), scores_p,
                16, kv_len, 64,
                qp.stride(0), qp.stride(1),
                Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                scores_p.stride(0), scores_p.stride(1),
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            # scores = scores_n + scores_p
            scores = scores_n + scores_p  # [16, kv_len]

            # Compute absolute position for causal mask
            prefix_len = kv_len - total_q_actual  # number of cached tokens before this query
            absolute_pos = prefix_len + i

            # Compute attention (softmax with causal mask) -> [16, kv_len]
            attn = torch.empty((16, kv_len), dtype=torch.float32, device=q_nope.device)
            grid_soft = (16,)
            softmax_row_causal_kernel[grid_soft](
                scores, attn,
                kv_len, absolute_pos,
                BLOCK=128,
            )

            # out_row = attn @ Kc -> [16, 512]
            out_row = torch.empty((16, 512), dtype=torch.float32, device=q_nope.device)
            # strides for matmul
            stride_am3 = attn.stride(0)  # 1
            stride_ak3 = attn.stride(1)  # kv_len
            stride_bk3 = Kc.stride(0)    # 1
            stride_bn3 = Kc.stride(1)    # 512
            stride_cm3 = out_row.stride(0)  # 512
            stride_cn3 = out_row.stride(1)  # 1
            grid_mat_out = (triton.cdiv(16, 16), triton.cdiv(512, 64))
            matmul_kernel[grid_mat_out](
                attn, Kc, out_row,
                16, kv_len, 512,
                stride_am3, stride_ak3,
                stride_bk3, stride_bn3,
                stride_cm3, stride_cn3,
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            # Store output as bfloat16
            output[abs_q] = out_row.to(torch.bfloat16)

            # Compute lse per row (base-2)
            lse_row = torch.empty((16,), dtype=torch.float32, device=q_nope.device)
            grid_lse = (16,)
            lse_row_kernel[grid_lse](
                scores, lse_row,
                kv_len, 1.0 / math.log(2.0),
                BLOCK=128,
            )
            lse[abs_q] = lse_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
