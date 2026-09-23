import torch
import triton
import triton.language as tl

# 1D specialized kernel for M == 1: computes C[0, n:n+BLOCK_N] = sum_k A[0, k] * B_T[k, n:n+BLOCK_N]
@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # B is [N, K], we index as B_T[k, n] = B[n, k] using these strides
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,  # columns tile
    BLOCK_K: tl.constexpr,  # reduction tile
):
    # Each program handles a block of columns
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for this block of columns
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A row segment (M==1)
        a = tl.load(A + 0 * stride_am + ks * stride_ak, mask=ks < K, other=0.0)  # [BLOCK_K]
        a = a.to(tl.float32)

        # Load B tile as [BLOCK_K, BLOCK_N]: B_T[k, cols] = B[cols, k]
        b_ptrs = B + cols[None, :] * stride_bn + ks[:, None] * stride_bk  # [BLOCK_K, BLOCK_N]
        mask_b = (ks[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)  # [BLOCK_K, BLOCK_N]
        b = b.to(tl.float32)

        # Outer product accumulation for this K-chunk: acc += sum_k a[k] * b[k, :]
        # Use reduction along axis=0
        acc += tl.sum(b * a[:, None], axis=0)

    # Store results to C[0, cols]
    tl.store(C + 0 * stride_cm + cols * stride_cn, acc, mask=cols < N)


# Generic 2D GEMM kernel for M > 1: C[m, n] = sum_k A[m, k] * B_T[k, n]
@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        km = k0 + tl.arange(0, BLOCK_K)  # reduction indices

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + km[None, :] * stride_ak
        mask_a = (rm[:, None] < M) & (km[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        a = a.to(tl.float32)

        # B tile as [BLOCK_K, BLOCK_N]: B_T[k, rn] = B[rn, k]
        b_ptrs = B + rn[None, :] * stride_bn + km[:, None] * stride_bk
        mask_b = (km[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        b = b.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure dtypes are float16 as in the original setup
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"
        # We only support CUDA for Triton; fallback not needed in this task since inputs are on GPU
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"

        M, K = A.shape
        N = B.shape[1]  # B is [N, K], C is [M, N]

        # Allocate output as float32 accumulator, then cast to original dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Specialized fast path for M=1
            BLOCK_N = 256
            BLOCK_K = 128
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic GEMM fallback for other M
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul default behavior
        return C.to(A.dtype)