import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16,  num_stages=6),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=16,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=16,  num_stages=6),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1024, 'BLOCK_K': 64}, num_warps=16,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1024, 'BLOCK_K': 128}, num_warps=16,  num_stages=6),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: row (M), col (K)
    stride_bk, stride_bn,       # B strides: row (K), col (N) -> emulate B.T via strides
    stride_cm, stride_cn,       # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id: each program handles a tile [BLOCK_M, BLOCK_N] of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        # B tile: [BLOCK_K, BLOCK_N], using B's original strides to emulate B.T
        B_tile_ptrs = B_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K)
        b_mask = ((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load tiles (with masks) and cast to fp32 for accumulation
        a = tl.load(A_tile_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]
        b = tl.load(B_tile_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]

        # Accumulate: acc += a @ b
        acc += tl.dot(a, b)

    # Store result to C (Triton will cast to C's dtype if needed)
    C_tile_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure inputs are CUDA tensors and 2D
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Make inputs contiguous for better memory access
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (same dtype as inputs)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Grid function for autotune: depends on meta BLOCK_M/BLOCK_N
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel (autotuned) from ModelNew.forward
        _matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),     # A strides for row (M) and col (K)
            B.stride(1), B.stride(0),     # B strides for col (N) and row (K) to emulate B.T
            C.stride(0), C.stride(1),     # C strides for row (M) and col (N)
        )

        return C


def run(*args):
    return ModelNew()(*args)
