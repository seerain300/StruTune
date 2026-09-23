import torch
import triton
import triton.language as tl

# Optimized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B[k, n] using B's original strides, no materialization.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128,  'BLOCK_K': 64},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_K': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_K': 128},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 512},  num_warps=8,  num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides for [M, K]
    stride_bk, stride_bn,    # B strides for [K, N]
    stride_ym, stride_yn,    # Y strides for [M, N]
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single program dimension over N tiles since M == 1
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k] vector
        A_row_ptr = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a = tl.load(A_row_ptr, mask=k_offsets < K, other=0.0).to(tl.float32)

        # Load B_T[n, k] tile: shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate dot product for this K chunk: sum over K -> vector of size BLOCK_N
        acc += tl.sum(b * a[None, :], axis=0)

    # Store results to Y[0, n]
    Y_row_ptr = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = n_offsets < N
    tl.store(Y_row_ptr, acc, mask=y_mask)


# Generic Triton GEMM for M > 1:
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides for [M, K]
    stride_bk, stride_bn,    # B strides for [K, N]
    stride_ym, stride_yn,    # Y strides for [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile logically as B_T [BLOCK_K, BLOCK_N]: B[k, n]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to Y[m, n]
    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton execution requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Output tensor in fp32 for robust accumulation; cast to fp16 at return
        Y = torch.empty((M, N), dtype=torch.float32, device=A.device)

        if M == 1:
            # Grid over N tiles; autotuner chooses BLOCK_N
            grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
            # Cast to fp16 to match original code's output dtype
            return Y.to(torch.float16)
        else:
            # Generic GEMM for M > 1
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
            _generic_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y.to(torch.float16)


def run(*args):
    return ModelNew()(*args)
