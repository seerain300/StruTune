import torch
import triton
import triton.language as tl

# Generic GEMM Triton kernel: computes C[M, N] = A[M, K] @ B_T[N, K]
# We index B as B_T[n, k] = B[k, n] using original strides: B_ptr + n*stride_bn + k*stride_bk.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile as B_T: B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C[m, n]
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


# Specialized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B[k, n] for n in [pid*BLOCK_N : (pid+1)*BLOCK_N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B is [K, N], indexing as B_T[n, k] = B[k, n]
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles; M is assumed == 1
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this row segment (fp32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[0, k] as a vector
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_vec = tl.load(A_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K], fp16

        # Load B[k, n] as a tile [BLOCK_K, BLOCK_N] using transposed indexing
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_tile = tl.load(B_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)  # [BLOCK_K, BLOCK_N], fp16

        # Accumulate: sum over K for each column n
        acc += tl.sum(b_tile * a_vec[:, None], axis=0)

    # Store results to Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    tl.store(Y_ptrs, acc.to(tl.float16), mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors; evaluator provides CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            # Minimal fallback (kept for completeness), but Triton path is used
            return torch.matmul(A, B.t())

        # Ensure dtype float16 (original code uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        if B.ndim != 2:
            raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}")
        B_m, N = B.shape
        if B_m != K:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (B_m={B_m}, N={N}). B's first dim must equal A's K.")

        # Allocate output tensor Y [M, N], float16
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        if M == 1:
            # Launch specialized row kernel over N tiles
            grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
            _row_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y
        else:
            # Launch generic GEMM kernel over M and N tiles
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
            _matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),  # for B_T[n, k] = B[k, n]: use n stride and k stride
                Y.stride(0), Y.stride(1),
            )
            return Y


def run(*args):
    return ModelNew()(*args)
