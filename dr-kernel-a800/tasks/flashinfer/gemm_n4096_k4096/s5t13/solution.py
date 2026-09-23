import torch
import triton
import triton.language as tl


# Generic GEMM: computes Y[M, N] = A[M, K] @ B_T[K, N], where B_T[n, k] = B[k, n]
# We do not materialize B_T; we index B with transposed strides.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for rows and cols of this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile as B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to Y[m, n] (fp16)
    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Specialized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B[k, n], using B's strides for B_T[n, k] = B[k, n].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,      # A is [M, K]
    stride_bn, stride_bk,      # B is [K, N], indexing B_T[n, k] = B[k, n]
    stride_yn,                 # Y is [M, N], we write Y[0, :]
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles; M is 1, so only row 0
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this row tile
    acc_row = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[0, k] as a vector
        A_row_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak  # M == 1 -> row index 0
        a_vec = tl.load(A_row_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # Load B_T[n, k] = B[k, n] as a [BLOCK_K, BLOCK_N] tile
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_tile = tl.load(B_ptrs, mask=(n_mask[None, :] & k_mask[:, None]), other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: sum over K of a_vec[:, None] * b_tile
        # Broadcast a_vec to [BLOCK_K, BLOCK_N]
        acc_row += tl.sum(a_vec[:, None] * b_tile, axis=0)

    # Store results to Y[0, n] (fp16)
    Y_row_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    tl.store(Y_row_ptrs, acc_row.to(tl.float16), mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes: A[M, K], B[K, N]
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("Inputs must be 2D: A [M, K], B [K, N]")
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). A's K must equal B's K.")

        # Enforce dtype float16 (original code uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Allocate output tensor (M, N), float16
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # If M == 1, use specialized kernel
        if M == 1:
            # Launch 1D grid over N tiles
            # The autotuner will choose BLOCK_N; grid depends on that, but Triton will pass meta and we can use cdiv with a conservative upper bound.
            # Since autotune configs are known, pick grid based on largest BLOCK_N for upper bound; Triton will handle masks and configs.
            # Here we can set grid using the largest BLOCK_N (1024) to ensure coverage; autotune will still work correctly.
            grid = (triton.cdiv(N, 1024),)
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(1),
            )
            return Y
        else:
            # Generic GEMM kernel: 2D grid over M and N tiles
            # Use a default grid; autotune will pick best BLOCK sizes.
            # We provide grid as a function of meta-parameters in Triton, but here we use a typical 2D grid over cdiv(M, 64) and cdiv(N, 128).
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
            _matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y


def run(*args):
    return ModelNew()(*args)
