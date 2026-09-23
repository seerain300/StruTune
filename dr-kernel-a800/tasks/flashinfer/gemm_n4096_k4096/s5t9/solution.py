import torch
import triton
import triton.language as tl


# 2D GEMM Triton kernel: computes C[M, N] = A[M, K] @ B_T[K, N] where B_T[n, k] = B[k, n]
@triton.jit
def _matmul_bt_kernel_2d(
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

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile: B_T[n, k] = B[k, n] -> use strides (stride_bk, stride_bn)
        # Shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


# Specialized Triton kernel for M == 1: computes Y[0, n] = sum_k A[0, k] * B[k, n]
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides
    stride_bk, stride_bn,   # B strides
    stride_ym, stride_yn,   # Y strides
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Only one row (M == 1), so grid is over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this tile across columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # A is a single row: A[0, k]
        A_vals = tl.load(A_ptr + 0 * stride_am + k_offsets * stride_ak, mask=(k_offsets < K), other=0.0)  # shape [BLOCK_K]

        # Load B_T tile for columns n_offsets: B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk  # shape [BLOCK_K, BLOCK_N]
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        B_vals = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate dot product across k-chunk
        # acc += sum_k A_vals[k] * B_vals[k, :]
        # Compute outer product and reduce along K-axis
        # Build a [1, BLOCK_K] tensor for A_vals to use tl.dot: broadcasting to [BLOCK_N, BLOCK_K] then reduce
        A_col = A_vals[:, None]  # [BLOCK_K, 1]
        partial = tl.dot(B_vals, A_col)  # [BLOCK_N, 1]
        acc += partial.squeeze(-1)

    # Store result to Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = (n_offsets < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure we are on CUDA for Triton execution
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Ensure dtype is float16 as in the original get_inputs
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}).")

        # Output tensor Y with shape [M, N]
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # If M == 1, use the specialized Triton kernel to compute per-column dot-products
        if M == 1:
            # Fixed tile sizes for robustness
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
            return Y

        # Otherwise, use the generic 2D GEMM Triton kernel
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_bt_kernel_2d[grid](
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
