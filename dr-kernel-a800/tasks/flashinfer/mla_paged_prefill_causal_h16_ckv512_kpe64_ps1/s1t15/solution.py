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
    # 2D tiling over output rows (M) and cols (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax per row with causal mask: positions j > (absolute_pos) -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # multiply scores by scale before softmax
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=0.0)

    # Apply causal mask
    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    # Stable softmax: subtract max, then exp and normalize
    m = tl.max(x, axis=0)
    x = x - m
    x = x * scale
    exp_x = tl.exp(x)
    s = tl.sum(exp_x, axis=0)
    out = exp_x / s

    tl.store(Out_ptr + row_id * N + offsets, out, mask=mask)


# Logsumexp per row with causal mask (base-2 normalization)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # multiply scores by scale before logsumexp
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=0.0)
    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    m = tl.max(x, axis=0)
    x = x - m
    x = x * scale
    exp_x = tl.exp(x)
    s = tl.sum(exp_x, axis=0)
    lse_val = tl.log(s) / tl.log(2.0)  # base-2 logsumexp
    tl.store(Out_ptr + row_id, lse_val)


def matmul_grid(M, N, BLOCK_M, BLOCK_N):
    return (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Triton requires CUDA tensors
        device = q_nope.device
        assert device.type == "cuda", "This implementation requires CUDA tensors."

        # Shapes: q_nope [T, 16, 512], q_pe [T, 16, 64], ckv_cache [num_pages, 1, 512], kpe_cache [num_pages, 1, 64]
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Number of batches
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Prepare outputs
        output = torch.empty((total_q, 16, 512), dtype=torch.float32, device=device)  # we'll cast to bf16 later
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV span
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue
            kv_len = page_end - page_beg

            # Gather K vectors for this batch
            tok_idx = kv_indices[page_beg:page_end]  # [kv_len] int32
            Kc = ckv_cache[tok_idx]  # [kv_len, 1, 512]
            Kp = kpe_cache[tok_idx]  # [kv_len, 1, 64]

            # Iterate queries
            for i in range(q_len):
                abs_q = q_start + i

                # scores_n = qn @ Kc.T -> [16, kv_len]
                qn_vec = q_nope[abs_q]  # [16, 512], bfloat16
                qn_f32 = qn_vec.to(torch.float32)  # [16, 512]
                Kc_f32 = Kc.to(torch.float32)      # [kv_len, 512]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_n = matmul_grid(16, kv_len, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
                matmul_kernel[grid_n](
                    qn_f32, Kc_f32, scores_n,
                    16, 512, kv_len,
                    qn_f32.stride(0), qn_f32.stride(1),
                    Kc_f32.stride(0), Kc_f32.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    16, 64, 64,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                qp_vec = q_pe[abs_q]    # [16, 64], bfloat16
                qp_f32 = qp_vec.to(torch.float32)  # [16, 64]
                Kp_f32 = Kp.to(torch.float32)      # [kv_len, 64]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_p = matmul_grid(16, kv_len, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
                matmul_kernel[grid_p](
                    qp_f32, Kp_f32, scores_p,
                    16, 64, kv_len,
                    qp_f32.stride(0), qp_f32.stride(1),
                    Kp_f32.stride(0), Kp_f32.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    16, 64, 64,
                    num_warps=4, num_stages=2
                )

                # Combine
                scores = scores_n + scores_p  # [16, kv_len]
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Softmax with causal mask
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len, num_warps=1, num_stages=1
                )

                # Output = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[matmul_grid(16, 512, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)](
                    attn, Kc_f32, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc_f32.stride(0), Kc_f32.stride(1),
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
                    BLOCK=kv_len, num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row

        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
