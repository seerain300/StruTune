import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for general GEMM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},   num_warps=2, num_stages=3),
        # Larger N tiles to reduce number of N tiles for big N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        # Even larger N tiles for very large N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        # Smaller tiles for small M/N
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},   num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_at_transposed_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for 2D tiling over C tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    A_tile_ptr = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_tile_ptr = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k + offs_k[:, None] < K)
        # Load tiles
        A_block = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_block = tl.load(B_tile_ptr, mask=b_mask, other=0.0)
        # Accumulate (dot)
        acc += tl.dot(A_block, B_block)
        # Advance pointers
        A_tile_ptr += BLOCK_K * stride_ak
        B_tile_ptr += BLOCK_K * stride_bk

    # Write back results to C, casting to output dtype
    C_tile_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # We assume C is float32 for numeric stability; if input is float16, we cast on host after kernel.
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are contiguous
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's last dim {Kb} must match A's last dim {K}"

        # Allocate output (fp32 accumulation, cast later to A.dtype)
        C = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Compute strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)
        stride_bk = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid: one program per output tile
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_at_transposed_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to A.dtype to match reference output dtype
        if C.dtype != A.dtype:
            C = C.to(A.dtype)

        return C


def run(*args):
    return ModelNew()(*args)
