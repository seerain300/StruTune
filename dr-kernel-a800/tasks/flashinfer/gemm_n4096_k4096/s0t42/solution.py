import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M: minimize masked lanes along M, maximize parallelism along N
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 1024, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 2048, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 1024, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        # Medium/general
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        # Larger M
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512,  'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Output tile pointers
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load B^T tile: treat B as [N, K], B_T[k, n] = B[n, k]
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store as fp16
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _run_triton(A, B):
    """
    Compute C = A @ B.T using Triton. No torch matmul in host code.
    A: [M, K], B: [N, K], output C: [M, N].
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    N = B.shape[0]  # B is [N, K]

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(1)  # B's second dim (K)
    stride_bn = B.stride(0)  # B's first dim (N)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # 2D grid over tiles of M and N
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_general_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew.forward expects CUDA tensors.")
        return _run_triton(A, B)


def run(*args):
    return ModelNew()(*args)
