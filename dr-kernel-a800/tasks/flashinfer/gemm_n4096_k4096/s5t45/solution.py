import torch
import triton
import triton.language as tl

# Generic Triton GEMM: Y = A @ B_T, A[M, K], B[K, N], Y[M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # B_T[n, k] = B[k, n]; use original B strides
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Specialized Triton kernel for M == 1:
# Computes Y[0, n:n+BLOCK_N] = sum_k A[0, k] * B[k, n] using B's original strides (B_T[n, k] = B[k, n]).
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single row M == 1
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator vector for this row tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k]
        A_row_ptr = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_mask = (k_offsets < K)
        a = tl.load(A_row_ptr, mask=a_mask, other=0.0)  # shape [BLOCK_K]

        # Load B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: outer product and sum over K dimension
        # acc += sum_k (a_k * b[k, :])
        # Implement as matrix multiply over (1, BLOCK_K) and (BLOCK_K, BLOCK_N)
        acc += tl.dot(a[:, None], b)[0, :]  # shape [BLOCK_N]

    # Store Y[0, n:n+BLOCK_N]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = (n_offsets < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if not on CUDA (evaluator uses CUDA)
        if not (A.is_cuda and B.is_cuda):
            return torch.matmul(A, B.T)

        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}).")

        # Output tensor fp16
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_ym, stride_yn = Y.stride(0), Y.stride(1)

        if M == 1:
            # Launch specialized kernel for single-row case
            def grid_row(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)

            _row_matmul_bt_kernel[grid_row](
                A, B, Y,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_ym, stride_yn,
            )
            return Y
        else:
            # Generic GEMM for M > 1
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

            _matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_ym, stride_yn,
            )
            return Y


def run(*args):
    return ModelNew()(*args)
