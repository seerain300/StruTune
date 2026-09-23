import torch
import triton
import triton.language as tl

# Triton kernel for M == 1: computes Y[0, n] = sum_k A[0, k] * B[k, n]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128,  'BLOCK_K': 128}, num_warps=2, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,          # A strides
    stride_bk, stride_bn,          # B strides (for B_T[n, k] = B[k, n], we use (k, n))
    stride_ym, stride_yn,          # Y strides
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single row M=1, grid over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A row tile: A[0, k]
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_mask = k_offsets < K
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B_T tile: B_T[n, k] = B[k, n], so address is k*stride_bk + n*stride_bn
        BT_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        mask_b = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(BT_ptrs, mask=mask_b, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc[n] += sum_k a[k] * b[k, n]
        acc += tl.sum(b * a[:, None], axis=0)

    # Store Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = n_offsets < N
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Triton generic GEMM for M > 1: C[M, N] = A[M, K] @ B_T[K, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,          # A strides
    stride_bk, stride_bn,          # B strides for B_T[n, k] = B[k, n]
    stride_cm, stride_cn,          # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
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
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load B_T tile [BLOCK_K, BLOCK_N]: B_T[n, k] = B[k, n]
        BT_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(BT_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    mask_c = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask_c)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Validate shapes: A[M, K], B[K, N]
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Output tensor (fp32 for accumulation), shape (M, N)
        Y = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Use Triton kernels only; no torch ops
        if M == 1:
            grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),       # A strides
                B.stride(0), B.stride(1),       # B strides for B_T[n, k] = B[k, n]
                Y.stride(0), Y.stride(1),
            )
            return Y.to(torch.float16)
        else:
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
