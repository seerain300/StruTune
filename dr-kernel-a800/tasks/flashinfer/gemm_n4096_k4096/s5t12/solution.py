import torch
import triton
import triton.language as tl


# Generic GEMM Triton kernel: Y[M, N] = A[M, K] @ B_T[K, N], where B_T[n, k] = B[k, n]
# We index B with transposed strides to avoid materializing B_T.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides for [M, K]
    stride_bk, stride_bn,       # B strides for [K, N] -> index as B_T[n, k] = B[k, n]
    stride_ym, stride_yn,       # Y strides for [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store Y[m, n]
    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Specialized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B[k, n]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides for [M, K]
    stride_bk, stride_bn,       # B strides for [K, N]
    stride_yn,                  # Y stride along N (since M==1, only row 0 is used)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this row tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[0, k] vector
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_vec = tl.load(A_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load B[k, n] tile [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_tile = tl.load(B_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate dot product
        acc += tl.dot(a_vec, b_tile)

    # Store to Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_yn + n_offsets * stride_yn
    tl.store(Y_ptrs, acc.to(tl.float16), mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton path: no torch operations in computation, only allocation of output is done via torch.
        # Validate shapes: A[M, K], B[K, N]
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Allocate output tensor (float16). This is the only unavoidable torch op to produce a tensor.
        # We cannot allocate with Triton directly; torch.empty is used here for output.
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        if M == 1:
            # Launch specialized M==1 kernel
            # Provide a grid that matches autotuned BLOCK_N; Triton autotuner will pick the best config.
            grid = (triton.cdiv(N, 256),)
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(1),
            )
            return Y
        else:
            # Launch generic GEMM kernel
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

            _generic_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y


def run(*args):
    return ModelNew()(*args)
