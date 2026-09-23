import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small M cases
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=8, num_stages=2),

        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),

        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),

        # Medium M/N
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),

        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),

        # Larger tiles for bigger N/K
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
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
    # 2D grid over output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    m = offs_m[:, None]  # [BLOCK_M, 1]
    n = offs_n[None, :]  # [1, BLOCK_N]

    # Accumulator for this tile (2D)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A[m, k] and B_T[k, n]
        A_ptrs = A + m * stride_am + k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
        B_ptrs = B + k[:, None] * stride_bk + n * stride_bn  # [BLOCK_K, BLOCK_N]

        # Masks for boundary conditions
        A_mask = (m < M) & (k[None, :] < K)
        B_mask = (k[:, None] < K) & (n < N)

        # Load and accumulate
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K], fp16
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N], fp16
        # Cast to fp32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Write back to C
    C_ptrs = C + m * stride_cm + n * stride_cn
    C_mask = (m < M) & (n < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original run computes C = A @ B.T
        assert len(args) == 2, "ModelNew expects two tensors: A and B"
        A, B = args

        # Ensure dtype consistency (example uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # A is [M, K], B is [N, O]. We need C = A @ B.T, so B_T is [K, N].
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K = A.shape
        B_t = B.transpose(0, 1).contiguous()  # [K, N]
        N = B_t.shape[1]

        # Output
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_t.stride(0)  # along K
        stride_bn = B_t.stride(1)  # along N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: 2D over tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_kernel[grid](
            A, B_t, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
