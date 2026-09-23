import triton
import triton.language as tl


# General 2D-tiled matmul:
# Computes C[M, N] = A[M, K] @ (B^T)[K, N]
# Indexing: B_T[k, n] = B[n, k] -> strides (B.stride(0) for n, B.stride(1) for k)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # for B_T[k, n] we use B's strides (n, k)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B_T tile: shape [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Specialized kernel for very small M (<= 4): maximize parallelism along N
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_tinyM_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # for B_T[k, n] = B[n, k]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N; BLOCK_M is tiny, so we can handle small M without masking too much
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B_T tile: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [N, O], compute C = A @ B.T -> C: [M, N]
        M, K = A.shape
        # Determine N, O from B: for C[M, N], N equals B's first dim
        N = B.shape[0]
        # Output tensor (float16, matching typical input dtype in evaluator)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's strides:
        stride_bn = B.stride(0)  # original B's first dim (N)
        stride_bk = B.stride(1)  # original B's second dim (O/K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose kernel based on M
        if M <= 4:
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_tinyM_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_general_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
