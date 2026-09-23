import torch
import triton
import triton.language as tl


# Specialized 1D kernel for very small M (e.g., M == 1). Parallelizes along N and
# loops over M and K inside the kernel to minimize masked work.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=16, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def _matmul_m1_kernel(
    A, B, C,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # fp32 accumulator for BLOCK_N columns
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over rows of A (M can be small; typical case M==1)
    m = 0
    while m < M:
        # Loop over K in chunks
        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load A[m, k:k+BLOCK_K] as vector (BLOCK_K,)
            a_ptrs = A + m * stride_am + offs_k * stride_ak
            a = tl.load(a_ptrs, mask=mask_k, other=0.0)

            # Load B[k:k+BLOCK_K, n:n+BLOCK_N] as [BLOCK_K, BLOCK_N]
            b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = mask_k[:, None] & mask_n[None, :]
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Accumulate dot: (BLOCK_K,) dot (BLOCK_K, BLOCK_N) -> (BLOCK_N,)
            acc += tl.sum(a[:, None] * b, axis=0)
            k += BLOCK_K

        # Store results for row m
        c_ptrs = C + m * stride_cm + offs_n * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=mask_n)
        m += 1


# Generic 2D-tiled kernel for other shapes.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=16, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_2d_kernel(
    A, B, C,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B_T tile: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
        b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = mask_n[None, :] & mask_k[:, None]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        k += BLOCK_K

    # Store
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args

        # Device check
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew requires CUDA tensors. Please move inputs to CUDA.")

        # Ensure contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A is [M, K], B is [N, K]
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"A and B must be 2D. Got A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}.")
        M, K = A.shape
        N, K2 = B.shape
        if K2 != K:
            raise ValueError(f"B must have second dim equal to A.shape[1] (K). Got K={K2}, expected {K}.")

        # Output tensor (fp16 as in the provided example)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B_T strides: k-axis stride is B.stride(1), n-axis stride is B.stride(0)
        stride_bk = B.stride(1)  # corresponds to original B's second dim
        stride_bn = B.stride(0)  # corresponds to original B's first dim
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # If M is very small (e.g., M <= 4), use the specialized 1D kernel along N.
        # This avoids heavy masking on M and keeps parallelism along N.
        if M <= 4:
            # Grid is 1D over N tiles; autotune will select BLOCK_N from configs
            grid = (triton.cdiv(N, 256),)  # initial grid; autotune adjusts
            _matmul_m1_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )
            return C

        # Otherwise, use the generic 2D kernel.
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        _matmul_2d_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
