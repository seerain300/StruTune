import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M/N or edge cases
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4,  num_stages=3),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_kernel(
    A_ptr,  # *A, shape [M, K]
    B_ptr,  # *B, shape [K, N] (note: B is as-is, we index it as B.T)
    C_ptr,  # *C, shape [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,  # strides for B: bk=0 stride (rows), bn=1 stride (cols)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-chunk for this (m, n) tile
    A_tile_ptr = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # For B, we want B^T indexing: B[k, n] = A[k, n] with strides (bk, bn)
    # So we load B as [BLOCK_K, BLOCK_N] block using k along bk and n along bn
    B_tile_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load A[m, k] chunk: [BLOCK_M, BLOCK_K]
        A_chunk = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        # Load B[k, n] chunk: [BLOCK_K, BLOCK_N] (indexing B as B.T via strides)
        B_chunk = tl.load(B_tile_ptr, mask=b_mask, other=0.0)

        # Accumulate: acc += A_chunk @ B_chunk
        acc += tl.dot(A_chunk, B_chunk)

        # Advance pointers for next K chunk
        A_tile_ptr += BLOCK_K * stride_ak
        B_tile_ptr += BLOCK_K * stride_bk

    # Write back results for this tile, with bounds mask
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store to C at positions (m, n)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    # Cast acc to output dtype (Triton will handle cast; we keep fp32 and let store cast)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure we run on CUDA with Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."
        # Ensure contiguous for coalesced memory access
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, {N}]"

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Compute strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B (shape [K, N]), strides are:
        stride_bk = B.stride(0)  # along K dimension
        stride_bn = B.stride(1)  # along N dimension
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch Triton kernel: 2D grid over tiles
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))  # base grid; autotune will override BLOCKs
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
