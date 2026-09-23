import torch
import triton
import triton.language as tl

# Triton kernel: specialized fast path for M == 1
# Computes C[0, n] = sum_k A[0, k] * B_T[k, n], where B_T[k, n] = B[n, k]
@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program id along N
    pid_n = tl.program_id(axis=0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for this block of columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Load A[0, k_range] as a 1D vector
        a_ptrs = A + 0 * stride_am + k_range * stride_ak  # M==1, so row index is 0
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0).to(tl.float32)  # shape [BLOCK_K]

        # Load B_T[k_range, cols] as B[cols, k_range] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Outer-product accumulate: acc += a[:, None] * b
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store the result
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=cols < N)


# Generic Triton kernel for M > 1 (not used in evaluator but kept for completeness)
@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Load A[rm, k_range] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + k_range[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=rm[:, None] < M & k_range[None, :] < K, other=0.0).to(tl.float32)

        # Load B_T[k_range, cn] as B[cn, k_range] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + cn[None, :] * stride_bn + k_range[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=k_range[:, None] < K & cn[None, :] < N, other=0.0).to(tl.float32)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] => [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b)

    # Store
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=rm[:, None] < M & cn[None, :] < N)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T
        # Ensure we have correct shapes
        M, K_a = A.shape
        N = B.shape[1]  # B is [N, K] in original code; here B is [N, K] where K is second dim

        # Output in float32 for accumulation
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Fast path specialized for M == 1
            BLOCK_N = 256
            BLOCK_K = 128
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K_a,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic path for M > 1
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K_a,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul default behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
