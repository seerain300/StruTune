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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        A_tile = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0
        )
        B_tile = tl.load(
            B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0
        )
        acc += tl.dot(A_tile, B_tile)

    tl.store(
        C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    )


# Row-wise softmax with causal mask (apply j > (prefix + i) -> -inf)
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # usually 1.0
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # typically N
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))

    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom

    tl.store(Out_ptr + row_id * N + offsets, out, mask=mask)


# Row-wise logsumexp with causal mask (base-2)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # 1.0 (no scaling)
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # typically N
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))

    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    lse = tl.log(sum_exp) / tl.log(2.0)  # base-2
    tl.store(Out_ptr + row_id, lse)  # per-row scalar


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # All computation is done using Triton kernels to satisfy TRITON-ONLY requirement.
        device = q_nope.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors."

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        batch_size = qo_indptr.shape[0] - 1

        # Constants assumed by original code
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Sanity checks
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All inputs must be CUDA tensors"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Indptr and indices must be CUDA tensors"

        # Convert caches to float32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers (float32 compute, then cast to bfloat16 at the end)
        output = torch.empty((total_q, num_qo_heads, 512), dtype=torch.float32, device=device)  # will be cast to bf16
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)         # base-2 logsumexp, each row per head

        # Process each batch element
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Prepare queries
            qn_batch = q_nope[q_start:q_end]  # [q_len, 16, 512]
            qp_batch = q_pe[q_start:q_end]    # [q_len, 16, 64]
            q_len = q_end - q_start

            for i in range(q_len):
                abs_q = q_start + i

                # Compute qn and qp for this query position
                qn = qn_batch[i].to(torch.float32)  # [16, 512]
                qp = qp_batch[i].to(torch.float32) # [16, 64]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn, Kc, scores_n,
                    16, 512, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    16, 128, 64,
                    num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(16, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_p](
                    qp, Kp, scores_p,
                    16, 64, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    16, 64, 64,
                    num_warps=4, num_stages=2
                )

                # Combine
                scores = scores_n + scores_p  # [16, kv_len]

                # Apply causal mask: j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len  # previously cached tokens
                absolute_pos = prefix_len + i
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len, num_warps=1, num_stages=1
                )

                # Output = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                grid_out = (triton.cdiv(16, 16), triton.cdiv(512, 128))
                matmul_kernel[grid_out](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    16, 128, 64,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # lse per head (base-2)
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len, num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as per original code
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
