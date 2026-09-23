import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Larger N tiles to reduce number of N tiles for big outputs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
        # Balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        # Smaller M/N scenarios
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        # Vary BLOCK_K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # strides for A: [M, K]
    stride_bn, stride_bk,   # strides for B: [N, K]
    stride_cm, stride_cn,   # strides for C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for 2D tiling over output C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A tile [BLOCK_M, BLOCK_K]
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Pointers for B tile [BLOCK_K, BLOCK_N], but we index B as [N, K] so we load B_T implicitly
    # We need B_T[k, n], which corresponds to B[n, k] in original layout. For loads we can address B[n, k]
    # and let the tile be [BLOCK_K, BLOCK_N] by how we construct the pointer:
    b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        # Masks to avoid out-of-bounds
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k0 + offs_k[:, None] < K)
        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
        # Advance pointers for next K-block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store results to C
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"Incompatible shapes: A is [M, {K}], B is [{N}, {Kb}]."
        # Allocate output in fp32 for numerical stability; cast to A.dtype at the end if needed
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        # Compute strides
        stride_am, stride_ak = A.stride()
        stride_bn, stride_bk = B.stride()
        stride_cm, stride_cn = C.stride()
        # Launch kernel with a dynamic grid function based on autotuned BLOCK sizes
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
        matmul_bt_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn)
        # Match original behavior: output dtype equals A.dtype
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
