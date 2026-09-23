import torch
import triton
import triton.language as tl

# Specialized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * BT[n, k], where BT is B.T materialized as [N, K].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr,      # *f16, A[M, K] with M==1
    BT_ptr,     # *f16, BT[N, K] (B.T contiguous)
    Y_ptr,      # *f16, Y[M, N] with M==1
    M, N, K,
    stride_am, stride_ak,         # strides for A (M==1 row)
    stride_bt_n, stride_bt_k,     # strides for BT
    stride_ym, stride_yn,         # strides for Y (M==1)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a tile along N of size BLOCK_N
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator (fp32) for this N tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A row segment: A[0, k] -> shape [BLOCK_K]
        A_row_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_mask = (k_offsets < K)
        a = tl.load(A_row_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load BT tile: BT[n, k] -> shape [BLOCK_N, BLOCK_K]
        BT_ptrs = BT_ptr + n_offsets[:, None] * stride_bt_n + k_offsets[None, :] * stride_bt_k
        bt_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate: acc[n] += sum_k a[k] * bt[n, k]
        acc += tl.sum(bt * a[None, :], axis=1)

    # Store results to Y[0, n]
    Y_row_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = (n_offsets < N)
    tl.store(Y_row_ptrs, acc.to(tl.float16), mask=y_mask)


# Generic 2D GEMM for M > 1 (kept for completeness)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B as B_T[n, k] = B[k, n]: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Ensure dtypes are float16 (original code uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        Bm, N = B.shape
        if K != Bm:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (Bm={Bm}, N={N}). B's K must equal A's K.")

        # Enforce contiguity for robust stride handling
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output tensor, contiguous, float16. Allocate as (M, N).
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        if M == 1:
            # Materialize BT = B.T contiguous as [N, K] for robust indexing
            BT = B_c.transpose(0, 1).contiguous()  # [N, K]
            # Launch Triton kernel over N tiles
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
            _row_matmul_bt_kernel[grid](
                A_c, BT, Y,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y
        else:
            # Generic 2D GEMM for M > 1
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            _generic_matmul_bt_kernel[grid](
                A_c, B_c, Y,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                B_c.stride(0), B_c.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y


def run(*args):
    return ModelNew()(*args)
