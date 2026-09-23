import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M/N
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_b_t_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id for tiling over C of shape [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Pointers for current K-chunk
        a_ptrs = A + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        b_ptrs = B + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        # Load B tile as if B.T were [K, N] -> [BLOCK_K, BLOCK_N]
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Convert to fp32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b)

    # Store results back to C as fp16 (cast from fp32 accumulator)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)  # Triton will cast to C's dtype if needed


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Inner dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"

        # Output tensor in fp16 to match original run's dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Launch Triton GEMM kernel over tiles of (M, N)
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
        matmul_b_t_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
