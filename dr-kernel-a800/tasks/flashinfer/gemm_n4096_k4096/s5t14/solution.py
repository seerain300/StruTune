import torch
import triton
import triton.language as tl


# Generic GEMM: computes Y[M, N] = A[M, K] @ B_T[K, N], where B_T[n, k] = B[k, n]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides: [M, K] -> am for rows, ak for cols
    stride_bn, stride_bk,    # B strides: [K, N] -> bk for K, bn for N
    stride_ym, stride_yn,    # Y strides: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[m, k] tile -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_T[n, k] = B[k, n] tile -> shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store Y[m, n]
    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Specialized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B[k, n]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 512}, num_warps=8, num_stages=5),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides: [M=1, K] -> am=0, ak=1 for contiguous
    stride_bn, stride_bk,    # B strides: [K, N] -> bk=B.stride(0), bn=B.stride(1)
    stride_yn,               # Y strides: [M=1, N] -> yn=Y.stride(1)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for this row (M==1)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k] as a vector [BLOCK_K]
        A_row_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        k_mask = k_offsets < K
        a_vec = tl.load(A_row_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load B_T[n, k] = B[k, n] as [BLOCK_K, BLOCK_N] tile
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        mask_b = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b_tile = tl.load(B_ptrs, mask=mask_b, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: sum over K chunk
        # a_vec: [BLOCK_K], b_tile: [BLOCK_K, BLOCK_N] => [BLOCK_N]
        acc += tl.sum(b_tile * a_vec[None, :], axis=0)

    # Store Y[0, n]
    Y_row_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn  # M==1, so stride_ym unused
    n_mask = n_offsets < N
    tl.store(Y_row_ptrs, acc.to(tl.float16), mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Shapes: A [M, K], B [K, N]
        M, K = A.shape
        K_B, N = B.shape
        if K != K_B:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_B}, N={N}).")

        # Ensure dtype is float16 for consistency with original code
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Output tensor Y [M, N], float16
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Launch appropriate Triton kernel
        if M == 1:
            # Use 1D grid over N tiles
            # grid depends on BLOCK_N selected by autotuner; provide a default; Triton will resolve per config
            grid = (triton.cdiv(N, 128),)
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(1),
            )
            return Y
        else:
            # Generic GEMM with 2D grid
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
            _gemm_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y


def run(*args):
    return ModelNew()(*args)
