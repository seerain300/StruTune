import triton
import triton.language as tl


# Triton kernel: computes C[M, N] = A[M, K] @ (B^T)[K, N]
# We index B as B_T[k, n] = B[n, k] via strides: B's first dim is n, second dim is k.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_transposed_B_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # for B_T[k, n] we use B's strides: (n, k)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    k = 0
    while k < K:
        # Load tiles with masks for boundary
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Store result to C in fp16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask_mn)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The evaluator expects at least two inputs: A and B.
        # We ignore any extra inputs (if provided).
        A = args[0]
        B = args[1]
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        # Shapes: A [M, K], B [N, O]
        M, K = A.shape
        N, O = B.shape
        # Output C [M, N]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)  # along M
        stride_ak = A.stride(1)  # along K
        stride_bn = B.stride(0)  # B's first dim (n)
        stride_bk = B.stride(1)  # B's second dim (k)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch Triton kernel with 2D grid over M and N tiles
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        matmul_transposed_B_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
