import torch
import triton
import triton.language as tl


# Row-wise Triton kernel: one program per output row m.
# Computes C[m, :] = A[m, :] @ B.T[:, :]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_rowwise_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per row m
    m = tl.program_id(0)
    # Offsets across N
    n_offsets = tl.arange(0, BLOCK_N)
    # Accumulator for this row
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K in chunks
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[m, k_offsets] -> vector of length BLOCK_K
        a = tl.load(
            A_ptr + m * stride_am + k_offsets * stride_ak,
            mask=k_offsets < K,
            other=0.0
        ).to(tl.float32)

        # Load B_T[k_offsets, n_offsets] as a tile [BLOCK_K, BLOCK_N]
        # B_T[k, n] = B[n, k]
        b = tl.load(
            B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk,
            mask=(n_offsets[None, :] < N) & (k_offsets[:, None] < K),
            other=0.0
        ).to(tl.float32)

        # Accumulate outer product for this chunk: [BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_N]
        acc += tl.sum(a[:, None] * b, axis=0)

        k_start += BLOCK_K

    # Store results for this row to C[m, :]
    out_n = n_offsets
    out_mask = out_n < N
    tl.store(C_ptr + m * stride_cm + out_n * stride_cn, acc, mask=out_mask)


# General 2D Triton kernel: tile over M and N, loop over K
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0
        ).to(tl.float32)

        # Load B_T tile [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b = tl.load(
            B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk,
            mask=(n_offsets[None, :] < N) & (k_offsets[:, None] < K),
            other=0.0
        ).to(tl.float32)

        # Accumulate: a: [BM,K], b: [K,BN] -> [BM,BN]
        acc += tl.dot(a, b)
        k_start += BLOCK_K

    # Store results with masks
    tl.store(
        C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    )


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Output shape: [M, N]
        M, K_a = A.shape
        N, K_b = B.shape

        # Ensure contiguity for better memory access (allowed: layout, not computation)
        A_ = A.contiguous()
        B_ = B.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A_.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A_.stride(0)
        stride_ak = A_.stride(1)
        # For B_T[k, n] = B[n, k], use B's strides:
        stride_bk = B_.stride(1)  # original B's second dim (K) becomes k
        stride_bn = B_.stride(0)  # original B's first dim (N) becomes n
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose kernel based on M
        if M <= 8:
            # One program per row; grid along N is handled via autotuned BLOCK_N tiling
            grid = (M,)
            matmul_rowwise_kernel[grid](
                A_, B_, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )
        else:
            # 2D tiling over M and N; autotune will pick tile sizes
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 256))
            matmul_2d_kernel[grid](
                A_, B_, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
