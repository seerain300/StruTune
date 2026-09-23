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
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A: [M, K], B: [K, N]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak  # [BM, BK]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn  # [BK, BN]

        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        acc += tl.dot(a, b)  # [BM, BN]

    # Write C: [M, N]
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


@triton.jit
def lse_row_triton(X_ptr, Out_ptr, N: tl.int32, absolute_pos: tl.int32, BLOCK: tl.constexpr):
    # Each program computes the LSE for one row
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    # Apply causal mask: positions j > (absolute_pos) -> -inf
    x = tl.where(idx > absolute_pos, -float("inf"), x)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) / tl.log(2.0)  # base-2
    tl.store(Out_ptr + row_id, lse)


@triton.jit
def softmax_row_triton(X_ptr, Out_ptr, N: tl.int32, absolute_pos: tl.int32, BLOCK: tl.constexpr):
    # Softmax with causal mask
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    x = tl.where(idx > absolute_pos, -float("inf"), x)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Out_ptr + row_id * N + idx, out, mask=idx < N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Triton tiling defaults (can be tuned)
        self.BLOCK_M = 64
        self.BLOCK_N = 128
        self.BLOCK_K = 32
        self.num_warps = 4
        self.num_stages = 2
        self.SM_SCALE = 1.0

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape

        # Checks (match original)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare caches: [num_pages, D]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV tokens for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query in this batch element
            for i in range(q_len):
                abs_q = q_start + i
                # Load q_nope and q_pe for this query, shape [16, D]
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)   # [16, 64]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(16, self.BLOCK_M), triton.cdiv(kv_len, self.BLOCK_N))
                matmul_kernel[grid_n](
                    qn, Kc.T, scores_n,
                    16, 512, kv_len,
                    1, 512,   # strides: A strides (q_len=16, D=512)
                    Kc.T.stride(1), Kc.T.stride(0),   # B strides: Kc.T [512, kv_len] -> (stride along N, stride along K)
                    scores_n.stride(1), scores_n.stride(0),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(16, self.BLOCK_M), triton.cdiv(kv_len, self.BLOCK_N))
                matmul_kernel[grid_p](
                    qp, Kp.T, scores_p,
                    16, 64, kv_len,
                    1, 64,    # A strides for qp
                    Kp.T.stride(1), Kp.T.stride(0),    # B strides for Kp.T
                    scores_p.stride(1), scores_p.stride(0),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )

                # Combine
                scores = scores_n + scores_p  # [16, kv_len]

                # Apply causal mask: j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i
                # Triton LSE
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                grid_lse = (16,)
                lse_row_triton[grid_lse](
                    scores, lse_row,
                    kv_len, absolute_pos, self.BLOCK_N
                )
                lse[abs_q] = lse_row  # [16]

                # Softmax and attention matmul
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid_softmax = (16,)
                softmax_row_triton[grid_softmax](
                    scores, attn,
                    kv_len, absolute_pos, self.BLOCK_N
                )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                grid_out = (triton.cdiv(16, self.BLOCK_M), triton.cdiv(512, self.BLOCK_N))
                matmul_kernel[grid_out](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(1), attn.stride(0),
                    Kc.stride(1), Kc.stride(0),
                    out_row.stride(1), out_row.stride(0),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )
                output[abs_q] = out_row  # float32
            # end for i
        # end for b

        # Cast output to bfloat16 as in original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
