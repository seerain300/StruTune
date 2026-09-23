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
    # 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Pointers for B tile: shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def lse_row_triton(X_ptr, Out_ptr, N: tl.int32, scale: tl.float32, absolute_pos: tl.int32, BLOCK: tl.constexpr):
    # Compute LSE for a single row of length N, with causal mask j <= absolute_pos
    # X_ptr points to [N], Out_ptr is scalar
    idx = tl.arange(0, BLOCK)
    # Load row with mask
    x = tl.load(X_ptr + idx, mask=idx < N, other=-float("inf"))
    # Apply causal mask: j > absolute_pos -> -inf
    x = tl.where(idx > absolute_pos, -float("inf"), x)
    # Stable LSE
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) / tl.log(2.0)  # base-2
    tl.store(Out_ptr, lse)


@triton.jit
def softmax_row_triton(X_ptr, Out_ptr, N: tl.int32, scale: tl.float32, absolute_pos: tl.int32, BLOCK: tl.constexpr):
    # Softmax for a single row of length N, with causal mask j <= absolute_pos, write to Out_ptr[N]
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + idx, mask=idx < N, other=-float("inf"))
    x = tl.where(idx > absolute_pos, -float("inf"), x)
    m = tl.max(x, axis=0)
    e = tl.exp((x - m) * scale)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Out_ptr + idx, out, mask=idx < N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Triton tiling defaults
        self.BLOCK_M = 16
        self.BLOCK_N = 64
        self.BLOCK_K = 32
        self.num_warps = 4
        self.num_stages = 2
        self.SM_SCALE = 1.0

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape

        # Checks
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare caches: [num_pages, 1, D] -> [num_pages, D]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # will cast to bf16
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # Determine token indices for this batch b from kv_indptr and kv_indices
            # tok_idx is the slice kv_indices[kv_indptr[b]:kv_indptr[b+1]]
            tok_idx_b = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())]  # [kv_len_b], int32
            Kc_batch = Kc_all[tok_idx_b]  # [kv_len_b, 512]
            Kp_batch = Kp_all[tok_idx_b]  # [kv_len_b, 64]

            # Prepare queries for this batch: [q_len, 16, 512] and [q_len, 16, 64]
            q_nope_batch = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32).contiguous()     # [q_len, 16, 64]

            for i in range(q_len):
                # Query vectors
                qn = q_nope_batch[i]  # [16, 512]
                qp = q_pe_batch[i]    # [16, 64]

                # Compute scores_n = qn @ Kc_batch.T, shape [16, kv_len_b]
                kv_len = Kc_batch.shape[0]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(1, triton.cdiv(kv_len, self.BLOCK_N),)](
                    qn, Kc_batch.transpose(0, 1).contiguous(), scores_n,
                    16, 512, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc_batch.transpose(0, 1).stride(0), Kc_batch.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )

                # Compute scores_p = qp @ Kp_batch.T, shape [16, kv_len_b]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(1, triton.cdiv(kv_len, self.BLOCK_N),)](
                    qp, Kp_batch.transpose(0, 1).contiguous(), scores_p,
                    16, 64, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp_batch.transpose(0, 1).stride(0), Kp_batch.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )

                # Combine
                scores = scores_n + scores_p  # [16, kv_len]

                # Apply causal mask: j > (prefix_len + i) -> -inf, where prefix_len = kv_len - q_len (from original paper)
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i
                # Triton lse and softmax
                scores_flat = scores.contiguous().view(16 * kv_len)  # [M*N], here M=16, N=kv_len
                # But lse_row_triton expects a row of length N, so we do it per row.
                # We can call lse for each row by reshaping: LSE per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                for h in range(16):
                    row = scores[h, :].contiguous()
                    lse_row[h] = lse_row_triton[(1,)](row, lse_row[h], kv_len, self.SM_SCALE, absolute_pos, BLOCK=kv_len, num_warps=1, num_stages=1)

                # Softmax per row
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                for h in range(16):
                    row = scores[h, :].contiguous()
                    out_row = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    softmax_row_triton[(1,)](row, out_row, kv_len, self.SM_SCALE, absolute_pos, BLOCK=kv_len, num_warps=1, num_stages=1)
                    attn[h, :] = out_row

                # Output: attn @ Kc_batch -> [16, 512]
                output_vec = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, self.BLOCK_M), triton.cdiv(512, self.BLOCK_N),)](
                    attn, Kc_batch, output_vec,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc_batch.stride(0), Kc_batch.stride(1),
                    output_vec.stride(0), output_vec.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )
                # Store to output[q_start + i]
                output[q_start + i] = output_vec.to(torch.bfloat16)
                # Store lse per head
                lse[q_start + i] = lse_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
