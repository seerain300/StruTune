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
    # 2D launch grid over M and N tiles
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

        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Softmax per row with causal mask (row-wise 1D, length N)
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, scale: tl.float32, absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N

    x = tl.load(X_ptr + row_id * N + idx, mask=mask, other=0.0)
    x = x * scale

    # Causal mask: positions j > absolute_pos -> -inf
    causal = idx > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    attn = e / s

    tl.store(Out_ptr + row_id * N + idx, attn, mask=mask)


# Row-wise logsumexp (base-2) with causal mask
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, scale: tl.float32, absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N

    x = tl.load(X_ptr + row_id * N + idx, mask=mask, other=0.0)
    x = x * scale

    causal = idx > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) / 1.4426950408889634  # ln(2)
    tl.store(Out_ptr + row_id, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        device = q_nope.device
        q_nope = q_nope.to(device, non_blocking=True)
        q_pe = q_pe.to(device, non_blocking=True)
        ckv_cache = ckv_cache.to(device, non_blocking=True)
        kpe_cache = kpe_cache.to(device, non_blocking=True)

        # Shapes
        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # As per original assertions
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Squeeze caches to [num_pages, 512] and [num_pages, 64]
        Kc_all = ckv_cache.squeeze(1)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate over batch b
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start
            # KV range for this batch
            if b + 1 >= kv_indptr.shape[0]:
                continue
            kv_len = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            # For each query token i in the batch
            for i in range(q_len):
                abs_q = q_start + i

                # Load qn, qp
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # A = qn, B = Kc.T
                A = qn
                B = Kc_all.transpose(0, 1)  # [512, kv_len]

                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    A, B, scores_n,
                    A.shape[0], A.shape[1], B.shape[1],
                    A.stride(0), A.stride(1),
                    B.stride(0), B.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    A.shape[1],
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                A2 = qp
                B2 = Kp_all.transpose(0, 1)  # [64, kv_len]

                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    A2, B2, scores_p,
                    A2.shape[0], A2.shape[1], B2.shape[1],
                    A2.stride(0), A2.stride(1),
                    B2.stride(0), B2.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    A2.shape[1],
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=2, num_stages=2
                )

                scores = scores_n + scores_p
                # Prefix len and absolute position for causal mask
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Softmax with causal mask: attn [16, kv_len]
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, sm_scale, absolute_pos,
                    BLOCK=kv_len, num_warps=2, num_stages=1
                )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                A3 = attn
                B3 = Kc_all  # [kv_len, 512]

                # Grid setup: tile along N=512 with 64 columns
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(512, 64),)](
                    A3, B3, out_row,
                    A3.shape[0], A3.shape[1], B3.shape[1],
                    A3.stride(0), A3.stride(1),
                    B3.stride(0), B3.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    A3.shape[1],
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # lse per head (base-2): compute per row
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, sm_scale, absolute_pos,
                    BLOCK=kv_len, num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as per original signature
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
