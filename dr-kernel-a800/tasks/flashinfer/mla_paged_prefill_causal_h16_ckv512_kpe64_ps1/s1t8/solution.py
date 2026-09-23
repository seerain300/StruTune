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

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    # Write back C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Softmax with causal mask: X [N], Out [N] per row
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,                 # 1.0 for no scaling
    absolute_pos: tl.int32,           # causal mask threshold
    BLOCK: tl.constexpr,              # BLOCK >= N
):
    row_id = tl.program_id(0)  # number of rows = number of queries
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=-float("inf"))

    # Apply causal mask: positions j > absolute_pos -> -inf
    j = tl.arange(0, BLOCK)
    mask_causal = j <= absolute_pos
    x = tl.where(mask_causal, x, -float("inf"))

    # Stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row_id * N + tl.arange(0, BLOCK), out, mask=tl.arange(0, BLOCK) < N)


# Logsumexp (base-2) with causal mask: X [N], Out [N] per row (float32)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,                 # 1.0
    absolute_pos: tl.int32,           # causal mask threshold
    BLOCK: tl.constexpr,              # BLOCK >= N
):
    row_id = tl.program_id(0)
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=-float("inf"))

    # Apply causal mask
    j = tl.arange(0, BLOCK)
    mask_causal = j <= absolute_pos
    x = tl.where(mask_causal, x, -float("inf"))

    # Stable logsumexp
    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    sum_e = tl.sum(e, axis=0)
    lse = tl.log(sum_e) / tl.log(2.0)  # base-2 logsumexp

    tl.store(Out_ptr + row_id * N + tl.arange(0, BLOCK), lse, mask=tl.arange(0, BLOCK) < N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        if device.type != 'cuda':
            raise RuntimeError("ModelNew requires CUDA tensors; move inputs to GPU.")

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        assert num_qo_heads == self.num_qo_heads, "num_qo_heads must be 16"

        # Prepare Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [num_pages, 64]

        output = torch.empty((total_q, num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        B = qo_indptr.numel() - 1
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            kv_len = kv_end - kv_start

            if q_len == 0 or kv_len == 0:
                continue

            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            for i in range(q_len):
                abs_q = q_start + i

                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]

                # scores_n = qn @ Kc.T  => [16, kv_len]
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64),)](
                    qn, Kc.T, scores_n,
                    num_qo_heads, 512, kv_len,
                    q_nope.stride(0), q_nope.stride(1),
                    Kc.T.stride(0), Kc.T.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T  => [16, kv_len]
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64),)](
                    qp, Kp.T, scores_p,
                    num_qo_heads, 64, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.T.stride(0), Kp.T.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # causal mask: j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i
                softmax_out = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(num_qo_heads,)](
                    scores, softmax_out,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len,
                    num_warps=1, num_stages=1
                )

                # out = softmax @ Kc  => [16, 512]
                out_row = torch.empty((num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(num_qo_heads, 16), triton.cdiv(self.head_dim_ckv, 64),)](
                    softmax_out, Kc, out_row,
                    num_qo_heads, kv_len, self.head_dim_ckv,
                    softmax_out.stride(0), softmax_out.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                output[abs_q] = out_row

                # lse per head
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len,
                    num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as original returns
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
