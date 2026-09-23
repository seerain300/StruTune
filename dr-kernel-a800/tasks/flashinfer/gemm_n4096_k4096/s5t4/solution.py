import torch
import triton
import triton.language as tl

# Generic 2D Triton kernel: computes Y = A @ B_T, where B_T[n, k] = B[k, n]
# A: [M, K], B: [K, N], Y: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,            # A strides: row (m), col (k)
    stride_bk, stride_bn,            # B strides: row (k), col (n) for original B[k, n]
    stride_ym, stride_yn,            # Y strides: row (m), col (n)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # B tile: B_T[n, k] = B[k, n]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store Y[m, n] as fp16
    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Specialized Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * BT[n, k], where BT is B.T contiguous [N, K].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 512}, num_warps=8, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, BT_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,            # A strides for [M=1, K]
    stride_bnt, stride_btk,          # BT strides for [N, K], contiguous => stride_bnt=1, stride_btk=K
    stride_ym, stride_yn,            # Y strides for [M=1, N]
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # We have M == 1, so grid is only over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for one row (M=1)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k] vector
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_mask = k_offsets < K
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_K]

        # Load BT[n, k] tile, BT is [N, K] contiguous
        BT_ptrs = BT_ptr + n_offsets[:, None] * stride_bnt + k_offsets[None, :] * stride_btk
        b_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        b = tl.load(BT_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_N, BLOCK_K]

        # Accumulate: dot(a, b) over K => (BLOCK_N,)
        acc += tl.sum(b * a[None, :], axis=1)

    # Store Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = n_offsets < N
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        if len(B.shape) != 2:
            raise ValueError(f"B must be 2D, got shape {B.shape}")
        if B.shape[0] != K:
            raise ValueError(f"B's first dimension must equal A's K ({K}), got {B.shape[0]}")
        N = B.shape[1]

        # Ensure dtype is float16 (original code uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # For M == 1: use specialized kernel with BT = B.T contiguous to ensure correctness
        if M == 1:
            # Make A and B contiguous for predictable strides
            A_c = A.contiguous()
            # Materialize BT = B.T as contiguous [N, K]
            BT = B.t().contiguous()

            # Output Y [1, N], contiguous float16
            Y = torch.empty((1, N), dtype=torch.float16, device=A.device).contiguous()

            # Grid depends on BLOCK_N; Triton autotuner picks config
            grid = (triton.cdiv(N, 512),)
            _row_matmul_bt_kernel[grid](
                A_c, BT, Y,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
            )
            # Return as [M, N]
            return Y

        # For general M > 1: robust 2D GEMM in Triton
        A_c = A.contiguous()
        B_c = B.contiguous()
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        # Launch 2D Triton kernel with grid derived from M,N
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        _matmul_bt_kernel[grid](
            A_c, B_c, Y,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            Y.stride(0), Y.stride(1),
        )
        return Y


def run(*args):
    return ModelNew()(*args)
