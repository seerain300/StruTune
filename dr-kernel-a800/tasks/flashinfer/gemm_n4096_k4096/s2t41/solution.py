import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M/N/K
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bT_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,          # strides for A [M, K]
    stride_bk, stride_bn,          # strides for B [K, N]
    stride_cm, stride_cn,          # strides for C [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        # Pointers for A and B tiles
        a_ptrs = A + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (promote to fp32 for accumulation)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results (fp16); C is fp16 in host code
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Inputs: A [M, K], B [K, N]
        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, "Incompatible shapes: A is [M, K], B is [K, N]"

        # Ensure contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        # Output in fp16 to match original run
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = C.stride()

        # Launch 2D grid; use grid(meta) so autotuner's BLOCK sizes determine grid
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_bT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
