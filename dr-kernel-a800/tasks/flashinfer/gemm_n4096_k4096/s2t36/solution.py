import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 非常小的 tiles 用於極端狹窄的 M
        triton.Config({'BLOCK_M': 8,   'BLOCK_N': 64,   'BLOCK_K': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,   'BLOCK_K': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,   'BLOCK_K': 64},  num_warps=4, num_stages=2),

        # 中等 tiles
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,   'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=4, num_stages=3),

        # 平衡 tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: (M, K)
    stride_bk, stride_bn,        # B strides: (K, N) -> emulate B.T by indexing (k, n)
    stride_cm, stride_cn,        # C strides: (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Tile pointers
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)    # A[m, k]
        B_tile_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)    # B[k, n] -> B.T indexed as (k, n)

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load and cast to fp32
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)
        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store back; Triton will cast to the element type of C_ptr if needed
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A {A.shape}, B {B.shape}"

        # Ensure contiguous for coalesced access
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output tensor (same dtype as inputs)
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Grid based on autotuned BLOCK sizes
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        # Launch Triton kernel
        matmul_bT_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            C.stride(0), C.stride(1),
        )
        return C


def run(*args):
    return ModelNew()(*args)
