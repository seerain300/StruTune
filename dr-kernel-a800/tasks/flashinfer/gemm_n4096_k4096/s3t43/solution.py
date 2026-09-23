import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_n, B_stride_k,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid of programs over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: (BLOCK_M, BLOCK_K), A is (M, K)
        A_tile_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)

        # Load B tile: (BLOCK_K, BLOCK_N), B is (N, K). We want B_T[k, n] = B[n, k], so pointer is B_ptr + n * B_stride_n + k * B_stride_k
        B_tile_ptrs = B_ptr + n_offsets[None, :] * B_stride_n + k_offsets[:, None] * B_stride_k
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)

        # Accumulate in float32
        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))

    # Store result to C (float16), with masking
    C_tile_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_tile_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and expected dtypes
        assert A.is_cuda and B.is_cuda, "Triton kernels require CUDA tensors."
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Expected float16 inputs for this Triton implementation."

        # Shapes: A is (M, K), B is (N, K), output C is (M, N)
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's second dimension must equal A's second dimension (got {Kb} vs {K})."

        # Make inputs contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Tiling parameters tuned for this environment
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return C


def run(*args):
    return ModelNew()(*args)
