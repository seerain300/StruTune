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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=16, num_stages=5),
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
    # 2D tiling over output matrix
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Tile pointers
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Load tiles (fp16), cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back to C (fp32)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Compute C = A @ B.T
        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Ensure contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor: accumulate in fp32 for numerical stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: one program per output tile
        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N, meta['BLOCK_N']),
            )

        matmul_at_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        # If strict dtype matching is required, cast here:
        # return C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
