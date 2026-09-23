import torch
import triton
import triton.language as tl


# 2D-tiled GEMM kernel: computes C[M, N] = A[M, K] @ B^T[K, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=6),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=6),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=7),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=16, num_stages=6),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=16, num_stages=6),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=16, num_stages=7),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: row (M), col (K)
    stride_bk, stride_bn,       # B strides: row (K), col (N) -> emulate B.T via strides
    stride_cm, stride_cn,       # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        A_tile_ptrs = A_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        B_tile_ptrs = B_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K)
        b_mask = ((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(A_tile_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(B_tile_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    C_tile_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=c_mask)


# 1D-tiled "row vector" kernel: iterate over M inside kernel, vectorize across N
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8,  num_stages=6),
        triton.Config({'BLOCK_N': 1024,'BLOCK_K': 128}, num_warps=16, num_stages=6),
        triton.Config({'BLOCK_N': 2048,'BLOCK_K': 128}, num_warps=16, num_stages=7),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=16, num_stages=6),
        triton.Config({'BLOCK_N': 1024,'BLOCK_K': 256}, num_warps=16, num_stages=7),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel_rowvec(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: row (M), col (K)
    stride_bk, stride_bn,       # B strides: row (K), col (N) -> emulate B.T via strides
    stride_cm, stride_cn,       # C strides: row (M), col (N)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a block of N columns; loop over all M rows
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize acc as [M, BLOCK_N] for all rows
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # For each row m, compute dot with B^T columns
        # We'll build a [1, BLOCK_K] vector from A[m, :] and accumulate into [1, BLOCK_N]
        for m in range(0, M):
            A_row_ptrs = A_ptr + m * stride_am + offs_k * stride_ak  # [BLOCK_K]
            B_tile_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

            # Masks
            a_mask = (offs_k < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

            # Load A[m, k0:k0+BLOCK_K] as vector [BLOCK_K]
            a_row = tl.load(A_row_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BK]

            # Load B[k0:k0+BLOCK_K, offs_n] as [BK, BN]
            b_vec = tl.load(B_tile_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]

            # Accumulate into acc[m, :]
            acc[m, :] += tl.sum(b_vec * a_row[:, None], axis=0)  # sum over K-chunk

    # Store results for all rows
    for m in range(0, M):
        C_row_ptrs = C_ptr + m * stride_cm + offs_n * stride_cn
        c_mask = (offs_n < N)
        tl.store(C_row_ptrs, acc[m, :], mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure inputs are CUDA tensors and shapes are 2D
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Make inputs contiguous to improve memory access
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (same dtype as inputs)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Select kernel based on shape heuristics:
        # If M is very small and N is large, use row-vector kernel to reduce idle threads.
        use_rowvec = (M <= 32) and (N >= 1024)

        if use_rowvec:
            # Grid across N tiles only
            def grid_rowvec(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)

            _matmul_bt_kernel_rowvec[grid_rowvec](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),
                C.stride(0), C.stride(1),
            )
        else:
            # 2D grid across M and N
            def grid_2d(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

            _matmul_bt_kernel_2d[grid_2d](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),
                C.stride(0), C.stride(1),
            )

        return C


def run(*args):
    return ModelNew()(*args)
