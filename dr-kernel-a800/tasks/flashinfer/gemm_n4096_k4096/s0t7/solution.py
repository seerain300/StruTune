import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 128,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,   'BLOCK_N': 128,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 256,  'BLOCK_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 2,   'BLOCK_N': 256,  'BLOCK_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 4,   'BLOCK_N': 128,  'BLOCK_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8,   'BLOCK_N': 128,  'BLOCK_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8,   'BLOCK_N': 256,  'BLOCK_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,     # A strides: row-major [M, K]
    stride_bk, stride_bn,     # B strides: row-major [K, N]  (we access B_T[k, n] = B[n, k])
    stride_cm, stride_cn,     # C strides: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        k_curr = k + offs_k  # [BLOCK_K]

        # A tile: [BLOCK_M, BLOCK_K], A[m, k] -> m along rows, k along cols
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_curr[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_curr[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # B_T tile: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + k_curr[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_curr[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result to C (cast to fp16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
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

        # Validate 2D shapes: A [M, K], B [K, N] to emulate B.T
        if A.dim() != 2:
            raise ValueError(f"A must be 2D, got shape {tuple(A.shape)}.")
        if B.dim() != 2:
            raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}.")
        M, K = A.shape
        K2, N = B.shape
        if K2 != K:
            raise ValueError(f"B must have second dim equal to A.shape[1] (K). Got K2={K2}, expected {K}.")

        # Output tensor (fp16 as in the provided example)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B is [K, N]; to emulate B_T[k, n] = B[n, k], we use:
        stride_bk = B.stride(0)  # original B's first dim (K)
        stride_bn = B.stride(1)  # original B's second dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid based on autotuned BLOCK_M and BLOCK_N
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        _matmul_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
