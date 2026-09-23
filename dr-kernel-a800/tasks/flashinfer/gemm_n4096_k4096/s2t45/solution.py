import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small tiles for tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 1024, 'BLOCK_K': 64}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1024, 'BLOCK_K': 64}, num_warps=16, num_stages=5),

        # Extreme N with tiny M
        triton.Config({'BLOCK_M': 8,   'BLOCK_N': 2048, 'BLOCK_K': 64}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 2048, 'BLOCK_K': 64}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 2048, 'BLOCK_K': 64}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 8,   'BLOCK_N': 4096, 'BLOCK_K': 64}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 4096, 'BLOCK_K': 64}, num_warps=16, num_stages=5),

        # Larger K cases
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_abt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # A: [M, K], B: [K, N]; emulate B.T by using B strides appropriately
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_abt(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B.T where A is [M, K], B is [K, N], returns [M, N].
    All computation is done by Triton; inputs must be CUDA tensors.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    if K != K_b:
        raise RuntimeError(f"Incompatible shapes: A is [M,{K}], B is [{K_b},N].")

    # Output tensor; keep dtype consistent with inputs
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Launch grid: one program per output tile
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_abt_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Triton-only computation of A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew expects two tensors (A, B).")
        A, B = args

        # Ensure tensors are on CUDA for Triton
        if not A.is_cuda:
            A = A.to('cuda')
        if not B.is_cuda:
            B = B.to('cuda')

        return triton_matmul_abt(A, B)


def run(*args):
    return ModelNew()(*args)
