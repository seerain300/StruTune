import math
import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# General 2D-tiled GEMM: C = A @ B.T
# A: [M, K], B: [N, K] (we read B[n, k] via strides to emulate B.T), C: [M, N]
@triton.jit
def _matmul_bt_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile: [BLOCK_K, BLOCK_N] by indexing B[n, k]
        # Note: B is [N, K] with strides (stride_bn, stride_bk). We want B_T[k, n] = B[n, k].
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = mask_n[None, :] & mask_k[:, None]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


# Row-wise kernel: one program per output row m
# Computes c[m, :] = A[m, :] @ B.T (i.e., dot product of A[m, :] with each column of B)
@triton.jit
def _matmul_bt_rowwise_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m = tl.program_id(axis=0)  # one program per row
    # Output vector pointer for row m
    # We don't know stride_cn in Triton here; assume C is row-major with stride 1 for columns.
    # Better: use a temporary [BLOCK_N] vector with pointers computed below.
    # Initialize output vector in fp32
    offs_n = tl.arange(0, BLOCK_N)
    acc_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A row slice [BLOCK_K]
        a_row_ptrs = A_ptr + (m * stride_am + offs_k * stride_ak)
        a_row_mask = mask_k
        a_row = tl.load(a_row_ptrs, mask=a_row_mask, other=0.0)  # [BLOCK_K]

        # Load B tiles [BLOCK_K, BLOCK_N] to accumulate into acc_vec
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
            b_mask = mask_n[None, :] & mask_k[:, None]
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

            # Accumulate: acc_vec += sum_k (A[m, k] * B[k, n]) for this block
            # Broadcast a_row [BLOCK_K] to [BLOCK_K, 1] and multiply with b_tile [BLOCK_K, BLOCK_N]
            acc_vec += tl.sum(b_tile * a_row[:, None], axis=0)

    # Store the row vector acc_vec
    # We need stride for columns; assume C is fp32 and contiguous in columns.
    # Store to C[m, :] with stride_cm on rows and stride 1 on columns (contiguous).
    c_row_ptrs = C_ptr + (m * stride_cm + offs_n)
    c_mask = offs_n < N
    tl.store(c_row_ptrs, acc_vec, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure Triton is available and tensors are on CUDA
        if not TRITON_AVAILABLE or not (A.is_cuda and B.is_cuda):
            # Fallback to PyTorch if Triton/CUDA not available (for robustness)
            return torch.matmul(A, B.T)

        # Validate shapes: A [M, K], B [N, K]
        assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
        M, K_a = A.shape
        N, K_b = B.shape
        assert K_a == K_b, "Inner dimensions must match for matmul"

        # Ensure contiguity (we'll use strides directly; contiguity helps performance)
        A = A.contiguous()
        B = B.contiguous()

        # We'll compute in fp32 for numerical stability and return fp32
        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose kernel based on M
        if M <= 32:
            # Row-wise kernel: one program per row
            BLOCK_N = 256  # vectorize along N
            BLOCK_K = 128  # reduction step
            grid = (M,)
            _matmul_bt_rowwise_kernel[grid](
                A, B, C,
                M, N, K_a,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2
            )
        else:
            # General 2D-tiled kernel
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 128
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_bt_kernel_2d[grid](
                A, B, C,
                M, N, K_a,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                NUM_WARPS=4, NUM_STAGES=4
            )

        # Return fp32 result (matches typical fp32 inputs in the provided setup)
        return C


def run(*args):
    return ModelNew()(*args)
