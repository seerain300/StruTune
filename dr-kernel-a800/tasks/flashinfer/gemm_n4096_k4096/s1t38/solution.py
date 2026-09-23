import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # General balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Larger N tiles for big-N cases
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Smaller K tiles for small-K cases
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
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
    # Program IDs for 2D tiling over output C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Pointers for A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile: [BLOCK_K, BLOCK_N] by indexing B[n, k]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store results to C[M, N]
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure we run on CUDA tensors; place on current device if needed
        # (get_inputs in the task likely provides CUDA tensors already)
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)

        # Expect shapes: A [M, K], B [N, K] -> C [M, N] = A @ B.T
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K = A.shape
        N, K_b = B.shape
        assert K_b == K, f"Incompatible shapes: A is [{M}, {K}], B is [{N}, {K_b}]"

        # Output tensor in fp32 for accumulation; cast to A.dtype after
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Make inputs contiguous for predictable strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Strides
        stride_am = A_c.stride(0)
        stride_ak = A_c.stride(1)
        stride_bn = B_c.stride(0)   # stride along N (rows of B)
        stride_bk = B_c.stride(1)   # stride along K (cols of B)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid: 2D over tiles of M and N
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        matmul_bt_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast output to match input A's dtype (typically float16)
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
