import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # General balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Larger N-tiles for better throughput when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        # Even larger N-tiles when shapes permit
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_atbT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Compute k indices for this block
        k_ids = k + offs_k

        # Pointers for A tile: A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (k_ids[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile: B[n, k] representing B^T[k, n] in the matmul
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (k_ids[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (k_ids[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)  # a: [BM, BK], b: [BK, BN]

    # Write back results
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_atbT(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton, with fp32 accumulation.
    A: [M, K], B: [N, K], returns C: [M, N], dtype matches A.dtype.
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"Incompatible shapes: A is [M, {K}], B is [{N}, {K_b}]"

    # Ensure contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    # Output in fp32 for accumulation, then cast to A.dtype for return
    C_fp32 = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Strides (in elements)
    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C_fp32.stride()

    # Grid is 2D over tiles; Triton autotune will select BLOCK sizes
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))  # initial grid; autotune overrides BLOCKs

    matmul_atbT_kernel[grid](
        A, B, C_fp32,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
    )
    # Cast to match input dtype
    C = C_fp32.to(A.dtype)
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two 2D tensors: A, B, compute C = A @ B.T using Triton
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B")
        A, B = args
        return triton_matmul_atbT(A, B)


def run(*args):
    return ModelNew()(*args)
