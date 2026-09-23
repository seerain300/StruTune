import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small N tiles for tiny M (e.g., M=1)
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 256, 'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Small M with moderate N tiles
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),

        # Larger N tiles for bigger matrices (N up to 8k+)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1024, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
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
    # Grid: 1D over rows (M), 2D over column tiles (N)
    pid_m = tl.program_id(0)  # row index
    pid_n = tl.program_id(1)  # tile index along N

    # Offsets for this program's row and N tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A: [M, K], B: [K, N]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (cast to fp32 for accumulation)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate using dot
        acc += tl.dot(a, b)

    # Write back to C (cast happens on store to C's dtype)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul_at(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], C: [M, N].
    All computation is performed inside Triton; tensors must be on CUDA.
    """
    assert A.is_cuda and B.is_cuda, "Triton kernel requires CUDA tensors"
    # Ensure contiguity for better memory access
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Output tensor (same dtype as inputs)
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Grid: 1D over M, 2D over N tiles
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_at_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1),
                           B.stride(0), B.stride(1), C.stride(0), C.stride(1))
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # ModelNew must invoke the Triton kernel; do not use torch.matmul in forward.
        assert len(args) == 2, "ModelNew expects two tensors: A and B"
        A, B = args
        return _triton_matmul_at(A, B)


def run(*args):
    return ModelNew()(*args)
