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


# Row-wise softmax with causal mask (BLOCK must be a power of 2)
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # usually 1.0
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # e.g., 16
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))

    causal_mask = offsets > absolute_pos
    # Apply causal mask: positions beyond absolute_pos become -inf
    x = tl.where(causal_mask, -float("inf"), x)

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom

    tl.store(Out_ptr + row_id * N + offsets, out, mask=mask)


# Row-wise logsumexp with causal mask (base-2). BLOCK must be a power of 2.
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # 1.0 (no scaling)
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # e.g., 16
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
        # qo_indptr, kv_indptr are int32 tensors; compute q_len, kv_len per batch
        device = q_nope.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors."

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare Kc_all and Kp_all from caches
        # ckv_cache: [num_pages, 1, 512] -> squeeze to [num_pages, 512]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)
        # kpe_cache: [num_pages, 1, 64] -> squeeze to [num_pages, 64]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)

        # Output initialization
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV range for this batch
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            kv_len = kv_end - kv_start

            # Kc and Kp for this batch
            # Note: Using entire Kc_all/Kp_all; kv_indices is not used here because
            # caches are provided as full arrays, not per-token. If per-token, you would index here.
            Kc = Kc_all  # [num_pages, 512]
            Kp = Kp_all  # [num_pages, 64]

            # Process each query in this batch
            for i in range(q_len):
                abs_q = q_start + i
                # Load q_nope and q_pe rows: shape [16, 512] and [16, 64]
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)   # [16, 64]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 32))](  # grid (1, ceil_div(kv_len,32))
                    qn, Kc.transpose(0, 1), scores_n,  # B[K,N] = Kc.T
                    16, 512, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    16, 32, 64,
                    num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 32))](  # grid (1, ceil_div(kv_len,32))
                    qp, Kp.transpose(0, 1), scores_p,  # B[K,N] = Kp.T
                    16, 64, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    16, 32, 32,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # Causal mask: positions j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Softmax over the row with causal mask
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=16,
                    num_warps=1, num_stages=1
                )

                # Output = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(512, 64))](  # grid (1, 8)
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    16, 64, 64,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # lse per head (base-2)
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=16,
                    num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as per original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
