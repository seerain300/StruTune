import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for general GEMM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        # Larger BLOCK_N to reduce N tiles for wide matrices
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        # Different BLOCK_K to reduce loop iterations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A tile [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # Pointers for B as if transposed: we need [BLOCK_K, BLOCK_N] but B is [N, K]
    # Addressing: b[n, k] => n*stride_bn + k*stride_bk, so we build a [BLOCK_K, BLOCK_N] pointer grid.
    B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

        # Advance to next K tile
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Write back to C
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # If tensors are not on CUDA, fall back to PyTorch
        if (not A.is_cuda) or (not B.is_cuda):
            return torch.matmul(A, B.T)

        M, K = A.shape
        N_b, K_b = B.shape
        assert K_b == K, f"Incompatible shapes: A is [{M}, {K}], B is [{N_b}, {K_b}]"

        # Output tensor in fp32 for accumulation, cast later
        out = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        # Ensure inputs are contiguous for predictable strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Strides
        stride_am = A_c.stride(0)
        stride_ak = A_c.stride(1)
        stride_bn = B_c.stride(0)  # stride along N (rows of B)
        stride_bk = B_c.stride(1)  # stride along K (cols of B)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        # Grid based on selected tile sizes
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N_b, META['BLOCK_N']),
        )

        matmul_bt_kernel[grid](
            A_c, B_c, out,
            M, N_b, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to input dtype to match expected output
        if out.dtype != A.dtype:
            out = out.to(A.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
