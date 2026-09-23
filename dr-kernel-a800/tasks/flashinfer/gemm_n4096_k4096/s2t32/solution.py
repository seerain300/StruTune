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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for 2D tiling over output matrix C of shape [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k in range(0, K, BLOCK_K):
        # Masks for boundary conditions
        k_mask = k + offs_k < K
        m_mask = offs_m < M
        n_mask = offs_n < N

        # Pointers for A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am) + ((k + offs_k)[None, :] * stride_ak)
        a_mask = (m_mask[:, None]) & (k_mask[None, :])
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # dtype follows A_ptr

        # Pointers for B tile as if B^T: we want B[k, n], but we access B using strides (k*stride_bk + n*stride_bn)
        # Here B tile is of shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        b_mask = (k_mask[:, None]) & (n_mask[None, :])
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)  # dtype follows B_ptr

        # Cast to fp32 for accumulation (inputs are typically fp16 in harness)
        A_tile_f32 = A_tile.to(tl.float32)
        B_tile_f32 = B_tile.to(tl.float32)

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(A_tile_f32, B_tile_f32)

    # Store results to C, casting back to output dtype (here we assume fp16 output)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure device is CUDA and tensors are contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [M, K]={A.shape}, B is [Kb, N]={B.shape}"

        # Allocate output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Compute grid based on tiling
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel
        matmul_at_transpose_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        return C


def run(*args):
    return ModelNew()(*args)
