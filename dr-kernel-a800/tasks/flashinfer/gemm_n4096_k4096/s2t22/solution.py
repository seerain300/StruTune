import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program IDs: each program computes a [BLOCK_M, BLOCK_N] tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Pointers for current A tile: shape [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        # Pointers for current B tile: we want B[k, n] using B's strides (B is [K, N])
        B_tile_ptr = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)

        # Cast to fp32 for accumulation
        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store result to C, with mask for boundaries
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure A is [M, K] and B is [K, N]
        assert A.ndim == 2 and B.ndim == 2, "A must be [M, K], B must be [K, N]"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]"

        # Match dtype and make contiguous
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (fp16 to match typical input dtype)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)   # along K dimension
        stride_bn = B.stride(1)   # along N dimension
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid sized according to META to cover full MxN for each autotune config
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        matmul_at_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
