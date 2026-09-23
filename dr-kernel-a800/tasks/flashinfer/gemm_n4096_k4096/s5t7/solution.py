import torch
import triton
import triton.language as tl


# Generic 2D GEMM Triton kernel: computes C[M, N] = A[M, K] @ B_T[K, N], with B_T[n, k] = B[k, n]
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: A[i, j] = A_ptr + i*stride_am + j*stride_ak
    stride_bk, stride_bn,   # B strides: B[i, j] = B_ptr + i*stride_bk + j*stride_bn
    stride_cm, stride_cn,   # C strides: C[i, j] = C_ptr + i*stride_cm + j*stride_cn
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile: B_T[n, k] = B[k, n], so index with (stride_bk, stride_bn)
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C[m, n] (float16)
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


# Specialized Triton kernel for M == 1: computes Y[0, n] = sum_k A[0, k] * B_T[n, k]
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides
    stride_bk, stride_bn,   # B strides
    stride_ym, stride_yn,   # Y strides
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Only one row in M == 1
    m = 0

    # Program id over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this tile of columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A[m, k] where m=0
        A_row_ptrs = A_ptr + m * stride_am + k_offsets * stride_ak
        a = tl.load(A_row_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk
        b = tl.load(B_ptrs, mask=(n_mask[:, None] & k_mask[None, :]), other=0.0)  # shape [BLOCK_N, BLOCK_K]

        # Accumulate dot product for each column in this tile: sum over K
        acc += tl.sum(b * a[None, :], axis=1)

    # Store results to Y[m, n]
    Y_row_ptrs = Y_ptr + m * stride_ym + n_offsets * stride_yn
    tl.store(Y_row_ptrs, acc.to(tl.float16), mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure Triton execution: inputs must be CUDA tensors and float16
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        B_M, N = B.shape
        if B_M != K:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (M={B_M}, N={N}). B's first dimension must equal A's K.")
        if M <= 0 or N <= 0:
            raise ValueError(f"Invalid dimensions: M={M}, N={N} must be positive.")

        # Output tensor Y [M, N]
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Choose kernels:
        if M == 1:
            # Specialized M==1 kernel: grid over N tiles
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
        else:
            # Generic 2D GEMM
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )

        return Y


def run(*args):
    return ModelNew()(*args)
