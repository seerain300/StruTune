import math
import torch
import triton
import triton.language as tl


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

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Compute pointers
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        # Load tiles
        A_tile = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        B_tile = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write back
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load row
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=True, other=-float("inf"))
    # Compute stable softmax with causal mask
    m = tl.max(x, axis=0)
    y = x - m
    e = tl.exp(y)
    # causal mask: positions > absolute_pos -> 0
    offs = tl.arange(0, BLOCK)
    mask_valid = offs <= absolute_pos
    e = tl.where(mask_valid, e, 0.0)
    sum_e = tl.sum(e, axis=0)
    soft = e / sum_e
    tl.store(Out_ptr + row_id * N + offs, soft, mask=(offs < N))


@triton.jit
def lse_row_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=True, other=0.0)
    m = tl.max(x, axis=0)
    y = x - m
    e = tl.exp(y)
    sum_e = tl.sum(e, axis=0)
    lse = tl.log(sum_e) * scale
    tl.store(Out_ptr + row_id, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Device and dtype
        device = q_nope.device

        # Convert caches to float32 and make contiguous
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        # qo_indptr and kv_indptr are int32 tensors of length 2 (batch=1)
        assert qo_indptr.shape[0] == 2 and kv_indptr.shape[0] == 2, "len_indptr must be 2 for this model."
        q_start = int(qo_indptr[0].item())
        q_end = int(qo_indptr[1].item())
        total_q = q_end - q_start

        # Compute tok indices for this batch
        kv_len = int(kv_indptr[1].item()) - int(kv_indptr[0].item())
        tok_idx = kv_indices[kv_indptr[0]: kv_indptr[1]].to(torch.int64)  # indices in [0, num_pages)
        Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
        Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

        # Output buffers
        output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Precompute scale for base-2 logsumexp
        ln2 = math.log(2.0)

        # Loop over each query position
        for i in range(total_q):
            abs_q = q_start + i

            # Load qn and qp: [16, 512] and [16, 64]
            qn = q_nope[abs_q].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[abs_q].contiguous().to(torch.float32)   # [16, 64]

            # scores_n = qn @ Kc.T -> [16, kv_len]
            scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
            grid_mat_n = (triton.cdiv(16, 16), triton.cdiv(kv_len, 128))
            matmul_kernel[grid_mat_n](
                qn, Kc.transpose(0, 1).contiguous(), scores_n,
                16, kv_len, 512,
                qn.stride(0), qn.stride(1),
                Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                scores_n.stride(0), scores_n.stride(1),
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            # scores_p = qp @ Kp.T -> [16, kv_len]
            scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
            grid_mat_p = (triton.cdiv(16, 16), triton.cdiv(kv_len, 128))
            matmul_kernel[grid_mat_p](
                qp, Kp.transpose(0, 1).contiguous(), scores_p,
                16, kv_len, 64,
                qp.stride(0), qp.stride(1),
                Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                scores_p.stride(0), scores_p.stride(1),
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )

            scores = scores_n + scores_p  # [16, kv_len]

            # Softmax (stable) with causal mask; grid=(16,)
            out_attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
            grid_soft = (16,)
            softmax_row_causal_kernel[grid_soft](
                scores, out_attn,
                kv_len,
                (kv_len - (total_q - 1)) + i,  # absolute_pos per query
                BLOCK=128,
            )

            # output row = attn @ Kc -> [16, 512]
            out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
            grid_mat_out = (triton.cdiv(16, 16), triton.cdiv(512, 128))
            matmul_kernel[grid_mat_out](
                out_attn, Kc, out_row,
                16, kv_len, 512,
                out_attn.stride(0), out_attn.stride(1),
                Kc.stride(0), Kc.stride(1),
                out_row.stride(0), out_row.stride(1),
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
                num_warps=4, num_stages=2,
            )
            output[i] = out_row.to(torch.bfloat16)

            # lse per head (base-2): grid=(16,)
            lse_row = torch.empty((16,), dtype=torch.float32, device=device)
            grid_lse = (16,)
            lse_row_kernel[grid_lse](
                scores, lse_row,
                kv_len,
                1.0 / ln2,
                BLOCK=128,
            )
            lse[i] = lse_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
