import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small M, large N, moderate K
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),

        # Medium M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),

        # Larger M
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Pointers for A tile: A[offs_m, k + offs_k]
        a_ptrs = A + (offs_m[:, None] * stride_am) + ((k + offs_k[None, :]) * stride_ak)
        # Pointers for B tile: B_T[k + offs_k, offs_n] where B_T[k, n] = B[n, k]
        b_ptrs = B + ((k + offs_k[:, None]) * stride_bk) + (offs_n[None, :] * stride_bn)

        # Masks to guard loads/stores for boundary tiles
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load tiles; cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store results back to C in fp16
    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def run_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton kernels only.
    A: [M, K], B: [N, K], C: [M, N]
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    N = B.shape[0]  # B is [N, K]

    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides (in elements)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(1)  # original B's second dim (K)
    stride_bn = B.stride(0)  # original B's first dim (N)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Launch grid: 2D over M and N tiles
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Only allocate, ensure contiguity, and launch Triton kernel; no torch matmul used.
        return run_triton(A, B)


def run(*args):
    return ModelNew()(*args)
