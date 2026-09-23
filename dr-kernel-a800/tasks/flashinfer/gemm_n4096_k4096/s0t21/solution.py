import triton
import triton.language as tl


# General 2D-tiled matmul kernel:
# Computes C[M, N] = A[M, K] @ (B^T)[K, N], where B^T is indexed via B[n, k] with strides.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # for B_T[k, n] = B[n, k], use B's strides (n, k)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A tile [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B tile interpreted as B_T[k, n] = B[n, k]
    B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k0 + offs_k[:, None] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


# Specialized kernel for very small M to reduce masked lanes and maximize parallelism along N.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k0 + offs_k[:, None] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous (data layout adjustment, not computation)
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A [M, K], B [N, O]; output C [M, N] = A @ B.T
        M = A.shape[0]
        N = B.shape[0]   # number of rows in B
        K = A.shape[1]   # second dimension of A (must match B's second dim for matmul)

        # Allocate output in fp16 to match original behavior
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's strides:
        stride_bn = B.stride(0)  # along n (rows of B)
        stride_bk = B.stride(1)  # along k (cols of B)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose and launch kernel
        if M <= 4:
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
            matmul_tinyM_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
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
