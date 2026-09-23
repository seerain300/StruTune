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
    # 2D grid over tiles of M and N
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


# Softmax with causal mask per row
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # scaling factor, usually 1.0
    absolute_pos: tl.int32,     # j > absolute_pos => -inf
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    x = x * scale
    # causal mask: set positions j > absolute_pos to -inf
    x = tl.where(idx <= absolute_pos, x, -float("inf"))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom
    tl.store(Out_ptr + row_id * N + idx, out, mask=idx < N)


# LogSumExp (base-2) with causal mask per row
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # scaling factor, usually 1.0
    absolute_pos: tl.int32,     # j > absolute_pos => -inf
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    x = x * scale
    x = tl.where(idx <= absolute_pos, x, -float("inf"))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    sum_exp = tl.sum(tl.exp(x), axis=0)
    ln2 = 0.6931471805599453  # log(2)
    out = tl.log(sum_exp) / ln2
    tl.store(Out_ptr + row_id, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"
        device = q_nope.device

        # Preprocess caches: keep as float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert kv_indices.dtype == torch.int32

        # Outputs
        output = torch.empty((total_q, 16, 512), dtype=torch.float32, device=device)  # bf16 at end
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Process batches
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV tokens for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            kv_len = tok_idx.numel()

            # Gather Kc_all and Kp_all
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # For each query in this batch
            for i in range(q_len):
                abs_q = q_start + i

                # qn: [16, 512], qp: [16, 64]
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]

                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn, Kc.T, scores_n,
                    16, 512, kv_len,
                    qn.stride(0), qn.stride(1),        # A strides
                    Kc.T.stride(0), Kc.T.stride(1),    # B strides
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=3
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(16, 16), triton.cdiv(kv_len, 32))
                matmul_kernel[grid_p](
                    qp, Kp.T, scores_p,
                    16, 64, kv_len,
                    qp.stride(0), qp.stride(1),        # A strides
                    Kp.T.stride(0), Kp.T.stride(1),    # B strides
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=3
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # Compute LSE per head (base-2) with causal mask
                for h in range(16):
                    lse_row = torch.empty((1,), dtype=torch.float32, device=device)
                    lse_row_causal_kernel[(1,)](
                        scores[h], lse_row,
                        kv_len,
                        sm_scale, (kv_len - q_len) + i,
                        BLOCK=kv_len
                    )
                    lse[abs_q, h] = lse_row[0]

                # Softmax attention per head and output
                for h in range(16):
                    scores_row = scores[h]  # [kv_len]
                    attn_row = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    softmax_row_causal_kernel[(1,)](
                        scores_row, attn_row,
                        kv_len,
                        sm_scale, (kv_len - q_len) + i,
                        BLOCK=kv_len
                    )
                    # out = attn @ Kc -> [512]
                    out_row = torch.empty((512,), dtype=torch.float32, device=device)
                    grid_out = (triton.cdiv(512, 64),)
                    matmul_kernel[grid_out](
                        attn_row[None, :], Kc, out_row[None, :],
                        1, kv_len, 512,
                        attn_row[None, :].stride(0), attn_row[None, :].stride(1),
                        Kc.stride(0), Kc.stride(1),
                        out_row[None, :].stride(0), out_row[None, :].stride(1),
                        BLOCK_M=1, BLOCK_N=64, BLOCK_K=64,
                        num_warps=1, num_stages=1
                    )
                    output[abs_q, h] = out_row

        # Cast output to bfloat16
        output = output.to(torch.bfloat16)
        return output, lse


# Optional compatibility: environments expecting 'Model'
class Model(ModelNew):
    pass


def run(*args):
    return ModelNew()(*args)
