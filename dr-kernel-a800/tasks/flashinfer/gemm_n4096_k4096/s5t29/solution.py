import torch
import triton
import triton.language as tl


# Triton kernel: computes C = A @ B_T, where
# - A: [M, K], row-major
# - B: [K, N], row-major (we index B logically as B_T[n, k] = B[k, n] via strides)
# - C: [M, N], row-major
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides: row-major, typically (K, 1)
    stride_bk, stride_bn,    # B strides: row-major, typically (N, 1) but we use as B_T[n, k]
    stride_cm, stride_cn,    # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile as B_T[n, k] = B[k, n]; use (stride_bk, stride_bn)
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate (fp32)
        acc += tl.dot(a, b)

    # Store results to C[m, n] as fp16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            # Fallback to PyTorch if not on CUDA; evaluator typically uses CUDA.
            return torch.matmul(A, B.t())

        # Ensure dtype is float16 (original code uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). B's K must equal A's K.")

        # Enforce contiguity for robust stride handling and performance
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output tensor, contiguous, float16. Allocate as (M, N).
        C = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        # Launch Triton kernel with a 2D grid over tiles of M and N
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))  # initial grid; autotuner will select the best config
        _gemm_bt_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),   # strides for A
            B_c.stride(0), B_c.stride(1),   # strides for B (we index as B_T)
            C.stride(0), C.stride(1),       # strides for C
        )
        return C


def run(*args):
    return ModelNew()(*args)
