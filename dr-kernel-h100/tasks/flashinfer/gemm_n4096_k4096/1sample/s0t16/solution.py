import torch
import triton
import triton.language as tl

# Specialized 1D kernel for M == 1: computes C[0, n] = sum_k A[0, k] * B.T[k, n]
@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # strides of B with shape (N, K)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program id over N columns
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for this block of columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A[0, k_range] (M==1)
        a_ptrs = A + 0 * stride_am + k_range * stride_ak  # shape [BLOCK_K]
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0)  # [BLOCK_K], fp16

        # Load B_T[k_range, cols] as B[cols, k_range] with shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        k_mask_1d = k_range < K  # [BLOCK_K]
        n_mask_1d = cols < N     # [BLOCK_N]
        b_mask = k_mask_1d[:, None] & n_mask_1d[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], fp16

        # Cast to fp32 for accumulation and multiply along K
        a32 = a.to(tl.float32)        # [BLOCK_K]
        b32 = b.to(tl.float32)        # [BLOCK_K, BLOCK_N]
        # Outer product accumulation: [BLOCK_N] += sum_k a32[k] * b32[k, :]
        acc += tl.sum(a32[:, None] * b32, axis=0)

    # Store results
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=cols < N)


# Generic 2D GEMM for C[m, n] = sum_k A[m, k] * B_T[k, n] (fallback when M > 1)
@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # strides of B with shape (N, K)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in C
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)  # reduction indices

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile: index as B[n, k] to form B_T[k, n]
        b_ptrs = B + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_mask = (rn[None, :] < N) & (rk[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: acc += A_tile @ B_tile^T
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store C
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA (Triton requires GPU)
        assert A.is_cuda and B.is_cuda, "Triton kernels require CUDA tensors"
        M, K_a = A.shape
        N, K_b = B.shape
        assert K_a == K_b, "A's second dim must equal B's first dim for A @ B.T"

        # Output in fp32 for accumulation accuracy; we will cast to A.dtype at the end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides of A, B, C
        stride_am, stride_ak = A.stride(0), A.stride(1)
        # B has shape (N, K) and we index it as B_T[k, n] = B[n, k]
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Fast path for M == 1 (dominant workload)
        if M == 1:
            BLOCK_N = 256
            BLOCK_K = 256  # reduce reduction iterations (K=4096 -> 16 iterations)
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic fallback for M > 1
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul default behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
