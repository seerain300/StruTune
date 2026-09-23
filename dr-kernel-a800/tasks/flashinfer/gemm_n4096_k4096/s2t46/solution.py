import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M or small N
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=2,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
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
    # 2D tiling: each program computes a tile [BLOCK_M, BLOCK_N] of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for the tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Runtime K loop to ensure full coverage
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load B tile corresponding to B.T: we want C[m, n] = sum_k A[m, k] * B[k, n]
        # In B's layout, element for k-th row and n-th col is at k*stride_bk + n*stride_bn
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Accumulate: acc += a @ b (elementwise multiply and sum over K axis)
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N] => a[:, :, None] * b[None, :, :] summed over K
        acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)

        k0 += BLOCK_K

    # Store the computed tile
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [K, N], output C: [M, N]
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        # Ensure contiguity for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Inner dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"

        # Output tensor in fp16 to match harness behavior
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid: one program per tile
        def grid(meta):
            BLOCK_M = meta['BLOCK_M']
            BLOCK_N = meta['BLOCK_N']
            return (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
