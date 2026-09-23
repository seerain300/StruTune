import torch
import triton
import triton.language as tl

# Autotuned Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B_T[n, k], where B_T is a contiguous [N, K] tensor.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128,  'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, BT_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bt_n, stride_bt_k,
    stride_yn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single row in A: M == 1
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k_offsets] -> shape [BLOCK_K], fp16
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_mask = (k_offsets < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT[n, k] -> shape [BLOCK_N, BLOCK_K], fp16
        BT_ptrs = BT_ptr + n_offsets[:, None] * stride_bt_n + k_offsets[None, :] * stride_bt_k
        bt_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate: acc[n] += sum_k a[k] * bt[n, k]
        acc += tl.sum(bt * a[None, :], axis=1)

    # Store Y[0, n_offsets] in fp16
    Y_ptrs = Y_ptr + n_offsets * stride_yn
    y_mask = (n_offsets < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


# Generic 2D GEMM for M > 1 (kept for completeness)
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, BT_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bt_n, stride_bt_k,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # A tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # BT tile [BLOCK_K, BLOCK_N]
        BT_ptrs = BT_ptr + n_offsets[None, :] * stride_bt_n + k_offsets[:, None] * stride_bt_k
        bt_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        acc += tl.dot(a, bt)

    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float16), mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Ensure dtype float16 (original code uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        N, Kb = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (N={N}, K={Kb}). B's K must equal A's K.")

        # Output tensor, float16, allocated as (M, N)
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # For M == 1, use the specialized Triton kernel with B_T materialized as a contiguous [N, K] tensor
        if M == 1:
            # Create B_T contiguous: [N, K]
            BT = B.t().contiguous()
            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)
            _row_matmul_bt_kernel[grid](
                A, BT, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(1),  # stride along N
            )
            return Y
        else:
            # Generic 2D kernel for M > 1
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            # Materialize B_T for the 2D kernel as well for robustness
            BT = B.t().contiguous()
            _generic_matmul_bt_kernel[grid](
                A, BT, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            return Y


def run(*args):
    return ModelNew()(*args)
