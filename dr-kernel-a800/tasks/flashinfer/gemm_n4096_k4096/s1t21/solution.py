import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for general GEMM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Smaller tiles for small M/N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Larger N tiles to reduce grid when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Even larger N for very wide B
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        # Very small M cases: tiny tiles
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Cases with very large K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K']  # autotune based on problem size
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # strides for A: [M, K]
    stride_bn, stride_bk,     # strides for B: [N, K]
    stride_cm, stride_cn,     # strides for C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D tiling over output C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)

        # Load B_T tile: treat B_T[k, n], where B[n, k] in original B
        # Pointer arithmetic: B_ptr + n * stride_bn + k * stride_bk
        B_tile_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        B_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_tile = tl.load(B_tile_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_tile @ B_tile
        acc += tl.dot(A_tile, B_tile)

    # Store result to C
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
        A = A.contiguous()
        B = B.contiguous()

        M, K_a = A.shape
        N_b, K_b = B.shape
        assert K_a == K_b, "A's second dimension (K) must equal B's first dimension (N) for C = A @ B.T."

        # Output in fp32 for numerical stability; cast later
        C_fp32 = torch.empty((M, N_b), dtype=torch.float32, device=A.device)

        # Strides in elements
        stride_am, stride_ak = A.stride()
        stride_bn, stride_bk = B.stride()  # B is [N, K]
        stride_cm, stride_cn = C_fp32.stride()

        # Use a grid function that depends on autotuned meta to match tile sizes
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_b, meta['BLOCK_N']))

        matmul_bt_kernel[grid](
            A, B, C_fp32,
            M, N_b, K_a,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast output to A.dtype to match input dtype
        C = C_fp32.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
