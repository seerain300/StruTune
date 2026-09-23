import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 1024}, num_warps=8,  num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_tinyM_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: tiles over M and N, with BLOCK_M=1 specialized
    pid_m = tl.program_id(0)  # for tinyM, M is 1, so grid_m = 1; kept for generality
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, 64):
        offs_k = k0 + tl.arange(0, 64)

        # A tile: A[m, k] for m in [BLOCK_M], typically BLOCK_M=1
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, 64]

        # B tile (transposed access): B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk  # [64, BLOCK_N]
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512},  num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, 64):
        offs_k = k0 + tl.arange(0, 64)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk  # B_T[k, n] = B[n, k]
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

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

        # Shapes: A is [M, K], B is [N, K] (note: B is not [N, O]; compute A @ B.T)
        M, K = A.shape
        if B.dim() != 2:
            raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}.")
        N, K2 = B.shape
        if K2 != K:
            raise ValueError(f"B must have second dim equal to A.shape[1] (K). Got K={K2}, expected {K}.")

        # Output tensor (fp16 as in the provided example)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's strides:
        stride_bk = B.stride(1)  # original B's second dim (K)
        stride_bn = B.stride(0)  # original B's first dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose kernel: specialize for tiny M (e.g., M <= 4)
        if M <= 4:
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_tinyM_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )
        else:
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_general_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
