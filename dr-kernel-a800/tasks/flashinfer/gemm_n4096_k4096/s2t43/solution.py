import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M/N
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),

        # Larger tiles for big K/N
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program id: each program handles a tile of size (BLOCK_M, BLOCK_N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_tile_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K], fp16
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], fp16

        # Accumulate
        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))  # [BLOCK_M, BLOCK_N]

    # Store results with mask
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2D and on CUDA; keep dtype as fp16 for output (inputs are fp16)
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        assert A.is_cuda and B.is_cuda, "A and B must be on CUDA device for Triton kernel"

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Make inputs contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        # Allocate output (fp32 accumulator, cast back to fp16 to match original)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Compute strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch Triton kernel
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))  # default grid; autotune will adjust performance
        gemm_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        # Cast to fp16 to match original behavior (inputs are fp16)
        C = C.to(torch.float16)
        return C


def run(*args):
    return ModelNew()(*args)
