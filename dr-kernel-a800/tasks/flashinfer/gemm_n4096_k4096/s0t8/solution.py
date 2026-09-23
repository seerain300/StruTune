import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,         # A strides
    stride_bk, stride_bn,         # B strides (this is B_T = transpose(B))
    stride_cm, stride_cn,         # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid: pid_m over M tiles, pid_n over N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers for A tile: [BLOCK_M, BLOCK_K]
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Create pointers for B_T tile: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (offs_k + k) < K  # [BLOCK_K] mask for current chunk
        a = tl.load(a_ptrs, mask=a_mask & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=b_mask & k_mask[:, None], other=0.0)
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result to C (fp16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original run expects two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args

        if not (A.is_cuda and B.is_cuda):
            # If not on CUDA, fall back to PyTorch to preserve correctness
            return torch.matmul(A, B.transpose(0, 1))

        # Ensure contiguous
        A = A.contiguous()
        # Form B_T with correct rank/strides without materializing data
        B_T = B.transpose(0, 1).contiguous()

        # Shapes: A is [M, K], B_T is [K, N]
        if A.dim() != 2:
            # Fallback if A is not 2D
            return torch.matmul(A, B_T)
        M, K = A.shape
        if B_T.dim() != 2:
            # Fallback if B_T is not 2D
            return torch.matmul(A, B_T)
        K2, N = B_T.shape
        if K2 != K:
            # Fallback if second dim of B_T doesn't match K of A
            return torch.matmul(A, B_T)

        # Output tensor (fp16 to match original dtype)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_T.stride(0)  # corresponds to original B's first dim (now k)
        stride_bn = B_T.stride(1)  # corresponds to original B's second dim (now n)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))  # initial grid; autotune will pick best config

        matmul_2d_kernel[grid](
            A, B_T, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
