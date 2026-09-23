import torch
import triton
import triton.language as tl

# Triton kernel for M == 1: computes C[0, n] = sum_k A[0, k] * B[k, n]
# Handles arbitrary strides for A and B; uses masks for partial tiles.
@triton.jit
def row_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per N tile
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this N tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A[0, k] vector of length BLOCK_K
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak  # m=0
        a_mask = (k_offsets < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_K]

        # Load B_T[n, k] tile of shape [BLOCK_N, BLOCK_K] using B strides
        B_ptrs = B_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk
        b_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_N, BLOCK_K]

        # Accumulate dot product: sum over k of a[k] * b[n, k]
        # acc[n] += sum_k (a[k] * b[n, k])
        acc += tl.sum(b * a[None, :], axis=1)

    # Store results to C[0, n]
    C_ptrs = C_ptr + 0 * stride_cm + n_offsets * stride_cn
    c_mask = (n_offsets < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


# Generic Triton GEMM for M > 1: C[M, N] = A[M, K] @ B_T[K, N]
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile [BLOCK_K, BLOCK_N] where B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C[m, n]
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Require CUDA and fp16
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}).")

        # Output tensor
        if M == 1:
            C = torch.empty((1, N), dtype=torch.float16, device=A.device)
            # Strides
            stride_am, stride_ak = A.stride(0), A.stride(1)
            stride_bk, stride_bn = B.stride(0), B.stride(1)
            stride_cm, stride_cn = C.stride(0), C.stride(1)

            # Choose tile sizes; no autotune to ensure robustness
            BLOCK_N = 256
            BLOCK_K = 64

            # Grid over N tiles
            grid = (triton.cdiv(N, BLOCK_N),)

            row_matmul_bt_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            return C

        else:
            # Generic path for M > 1
            C = torch.empty((M, N), dtype=torch.float16, device=A.device)

            stride_am, stride_ak = A.stride(0), A.stride(1)
            stride_bk, stride_bn = B.stride(0), B.stride(1)
            stride_cm, stride_cn = C.stride(0), C.stride(1)

            BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            matmul_bt_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            return C


def run(*args):
    return ModelNew()(*args)
