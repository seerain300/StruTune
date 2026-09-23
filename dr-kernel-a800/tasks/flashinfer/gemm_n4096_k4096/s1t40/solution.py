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
        # Specialized for small M (e.g., M=1): BLOCK_M=1
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Larger N tiles when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        # Larger K scenarios
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        # A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N], we need B[n, k] so index B_ptr + n*stride_bn + k*stride_bk
        b_ptrs = B_ptr + ((offs_n[None, :] * stride_bn) + (k + offs_k[:, None]) * stride_bk)
        b_mask = (offs_n[None, :] < N) & (k_mask[:, None])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_at_bT(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton.
    A: [M, K], B: [N, K], output C: [M, N].
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    assert A.shape[1] == B.shape[1], "A's K dimension must equal B's K dimension"
    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, "Incompatible shapes"

    # Make inputs contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    # Output in fp32 for accumulation precision, then cast to A.dtype
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bn = B.stride(0)
    stride_bk = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid based on chosen tile sizes (dynamic per autotune config)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    matmul_at_bT_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
    )

    # Cast to match A.dtype
    return C.to(A.dtype)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two input tensors: A and B")
        A, B = args
        return triton_matmul_at_bT(A, B)


def run(*args):
    return ModelNew()(*args)
