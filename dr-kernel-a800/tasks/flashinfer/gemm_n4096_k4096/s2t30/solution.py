import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small tiles for tiny M (e.g., M=1)
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K; higher warps/stages
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=16, num_stages=5),

        # Asymmetric tiles for very large N (e.g., N=8192) with tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 1024, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 1024, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 1024, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 2048, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 2048, 'BLOCK_K': 64},  num_warps=32, num_stages=5),
        triton.Config({'BLOCK_M': 1,    'BLOCK_N': 4096, 'BLOCK_K': 128}, num_warps=32, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute row/col offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        # A is [M, K], B is [K, N]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Write back to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors; Triton requires CUDA
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        # Ensure contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K_b, N = B.shape
        assert K == K_b, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Allocate output tensor in fp32 (compute precision)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = C.stride()

        # Launch grid
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        # Return C in fp32 (typical matmul precision for fp16 inputs).
        # If you need fp16 output, cast here: C = C.to(torch.float16)
        return C


def run(*args):
    return ModelNew()(*args)
