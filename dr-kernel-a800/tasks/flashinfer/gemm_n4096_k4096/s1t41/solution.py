import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for general GEMM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        # Favor larger N tiles when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Moderate M configs
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Larger K configs when K is big
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_at_b_t_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id for tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # For B, we want [BLOCK_K, BLOCK_N] and we index B with its original layout:
    # B[n, k] -> address = n*stride_bn + k*stride_bk
    B_tile_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k0 + offs_k[:, None] < K)

        # Load tiles, promote to fp32
        a = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)

        # Cast to fp32 for accumulation (inputs are fp16 here)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Fused multiply-add
        acc += tl.dot(a, b)

        # Advance pointers for next K-block
        A_tile_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk

    # Write back to C with boundary mask
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store fp32 accumulators
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"Incompatible shapes: A is (*, {K}), B is ({N}, {Kb})"

        # Allocate output (fp32 accumulator, will cast to A.dtype at the end)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Compute strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)
        stride_bk = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid: one program per output tile
        # Dynamic grid based on BLOCK sizes chosen by autotune
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        _matmul_at_b_t_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast output to input A dtype to match original behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
