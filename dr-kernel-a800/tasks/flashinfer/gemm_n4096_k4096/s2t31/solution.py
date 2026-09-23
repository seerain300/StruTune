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

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids over output C of shape [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers to A and B^T tiles
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
        BT_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)  # [BK, BN]

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles and cast to fp32 for accumulation
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]
        BT_tile = tl.load(BT_ptrs, mask=bt_mask, other=0.0).to(tl.float32)  # [BK, BN]

        # Accumulate using tl.dot
        acc += tl.dot(A_tile, BT_tile)  # [BM, BN], fp32

    # Store results (Triton will cast to output dtype if needed)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [K, N]
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Inner dimensions must match: A.shape[1]={K}, B.shape[0]={Kb}"

        # Ensure inputs are contiguous for coalesced access
        A = A.contiguous()
        # Materialize B^T as contiguous to avoid stride-based indexing issues
        BT = B.t().contiguous()

        # Output tensor (fp16, same as input dtype in harness)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_btk = BT.stride(0)  # row stride of BT
        stride_btn = BT.stride(1)  # col stride of BT
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_btk, stride_btn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
