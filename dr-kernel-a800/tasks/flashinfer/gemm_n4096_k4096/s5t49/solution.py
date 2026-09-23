import torch
import triton
import triton.language as tl

# Generic 2D GEMM: computes C[M, N] = A[M, K] @ B_T[K, N]
# B_T[n, k] is accessed via B's strides: (k_dim=K stride, n_dim=N stride).
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 1024,'BLOCK_K': 64},   num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _generic_matmul_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,      # B strides for (k, n) so B_T[n, k] -> B[k, n]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[m, k]
        A_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile: B_T[n, k] = B[k, n]
        B_ptrs = B + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C[m, n]
    C_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


# Special M==1 path: launch the generic kernel with BLOCK_M=1 and grid over N tiles
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024,'BLOCK_K': 64},   num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,      # B strides for (k, n) so B_T[n, k] = B[k, n]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # For M==1, this kernel is just the generic kernel with BLOCK_M=1 and grid over N
    pid_m = 0  # single row
    pid_n = tl.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # A[m, k] with m=0
        A_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # B_T[n, k] = B[k, n]
        B_ptrs = B + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    C_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs: Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        # Original code uses float16; ensure dtypes
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Shapes: A [M, K], B [K, N]
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). B's K must equal A's K.")

        # Output tensor, accumulate in fp32, return fp16
        Y = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Launch Triton kernel: if M==1, use row kernel; else use generic GEMM
        if M == 1:
            # Grid over N tiles; Triton autotuner selects BLOCK_N
            grid = (triton.cdiv(N, 1024),)  # placeholder; autotuner will override per-config
            _row_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )
        else:
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))  # placeholder; autotuner overrides
            _generic_matmul_bt_kernel[grid](
                A, B, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
            )

        # Cast to float16 to match original code's output dtype
        return Y.to(torch.float16)


def run(*args):
    return ModelNew()(*args)
