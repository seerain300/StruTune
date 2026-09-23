import torch
import triton
import triton.language as tl

# Specialized kernel for M == 1: compute C[0, :] = A[0, :] @ B_T[:].T
@triton.jit
def _row_matvec_kernel(
    A_ptr,        # *const T, shape [M=1, K]
    B_T_ptr,      # *const T, shape [K, N] (i.e., transpose of B: [N, K] -> [K, N])
    C_ptr,        # *float32, shape [1, N]
    M, N, K,      # int32
    stride_am, stride_ak,     # strides for A
    stride_bTr, stride_bTn,   # strides for B_T (row stride = original B's col stride, col stride = original B's row stride)
    stride_cm, stride_cn,     # strides for C
    BLOCK_N: tl.constexpr,    # columns tile
    BLOCK_K: tl.constexpr,    # reduction tile
):
    # Program id along N dimension
    pid_n = tl.program_id(axis=0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Masks for N boundary
    mask_n = cols < N

    # Accumulator for this block of columns (fp32)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_range < K

        # Load A[0, k] vector (1 row, BLOCK_K elements)
        a_vec = tl.load(A_ptr + 0 * stride_am + k_range * stride_ak, mask=mask_k, other=0.0)
        a_vec = a_vec.to(tl.float32)  # accumulate in fp32

        # Load B_T[k, cols] tile: shape [BLOCK_K, BLOCK_N]
        b_tile = tl.load(
            B_T_ptr + k_range[:, None] * stride_bTr + cols[None, :] * stride_bTn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        b_tile = b_tile.to(tl.float32)

        # Outer product accumulation: acc += sum_k a_vec[k] * b_tile[k, :]
        # Triton supports tl.sum across an axis
        acc += tl.sum(b_tile * a_vec[:, None], axis=0)

    # Store results to C[0, cols]
    tl.store(C_ptr + 0 * stride_cm + cols * stride_cn, acc, mask=mask_n)


# Generic 2D GEMM kernel: C[M, N] = A[M, K] @ B_T[K, N]
@triton.jit
def _matmul_generic_kernel(
    A_ptr, B_T_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid: over M tiles and N tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        # Load B_T tile: [BLOCK_K, BLOCK_N]
        b = tl.load(
            B_T_ptr + k_offsets[:, None] * stride_bTr + n_offsets[None, :] * stride_bTn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    tl.store(
        C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B.T using Triton. Ensure correctness by explicitly
        materializing B_T = B.transpose(0, 1) and passing its strides to the kernel.
        Accumulate in float32. Return tensor (dtype float32, matching expected numerical result).
        """
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K_a = A.shape
        N, K_b = B.shape
        # In the original code, B.T is used, so we must have K_a == K_b (same K).
        assert K_a == K_b, f"Incompatible shapes: A is [M, {K_a}], B is [{N}, {K_b}]"

        # Create B_T explicitly to guarantee correct shape and strides
        B_T = B.transpose(0, 1)  # [K, N]

        # Output tensor: shape [M, N], accumulate in float32 for stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B_T strides: if B has strides (b0, b1), then B_T has strides (b1, b0)
        stride_bTr = B.stride(1)
        stride_bTn = B.stride(0)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch specialized fast path for M == 1
        if M == 1:
            BLOCK_N = 256
            BLOCK_K = 128
            grid = (triton.cdiv(N, BLOCK_N),)
            _row_matvec_kernel[grid](
                A, B_T, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bTr, stride_bTn,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
            return C
        else:
            # Generic 2D GEMM
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_kernel[grid](
                A, B_T, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bTr, stride_bTn,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
            return C


def run(*args):
    return ModelNew()(*args)
