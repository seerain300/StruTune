import torch
import triton
import triton.language as tl


# Triton kernel for M == 1:
# Computes C[0, n] = sum_{k=0}^{K-1} A[0, k] * B_T[n, k], where B_T[n, k] = B[k, n].
# A is [1, K], B is [K, N], C is [1, N].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 512}, num_warps=8, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _matmul_row_bt_kernel(
    A_ptr,        # *fp16, shape [1, K]
    B_ptr,        # *fp16, shape [K, N]
    C_ptr,        # *fp16, shape [1, N]
    M: tl.constexpr,  # = 1
    N, K,
    stride_am, stride_ak,   # A strides
    stride_bk, stride_bn,   # B strides (for B[k, n], use (stride_bk, stride_bn))
    stride_cm, stride_cn,   # C strides
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program id over N tiles
    pid_n = tl.program_id(axis=0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # accumulator for this N tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # load A[0, k] vector chunk: shape (BLOCK_K,)
        a_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)

        # load B_T[n, k] tile: shape (BLOCK_N, BLOCK_K), using B strides for B[k, n]
        b_ptrs = B_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk
        b = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # dot: (BLOCK_N, BLOCK_K) @ (BLOCK_K,) -> (BLOCK_N,)
        # multiply and sum over K dimension
        acc += tl.sum(b * a[None, :], axis=1)

    # store results to C[0, n]
    c_ptrs = C_ptr + 0 * stride_cm + n_offsets * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=n_mask)


# Triton generic GEMM for M > 1:
# Computes C[M, N] = A[M, K] @ B_T[K, N], where B_T[n, k] = B[k, n].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_generic_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # B_T tile: [BLOCK_K, BLOCK_N], B_T[n, k] = B[k, n]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        acc += tl.dot(a, b)

    # store result
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors; ensure float16 dtype as in original
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
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). B's K must equal A's K.")

        # Make inputs contiguous to simplify stride handling (data movement, not compute)
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Allocate output tensor
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Launch appropriate Triton kernel
        if M == 1:
            # Grid over N tiles; autotuner will pick BLOCK_N; grid must match cdiv(N, BLOCK_N).
            # We don't know BLOCK_N here, so use a conservative upper bound; Triton handles cdiv internally.
            grid = (triton.cdiv(N, 256),)
            _matmul_row_bt_kernel[grid](
                A_c, B_c, C,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                B_c.stride(0), B_c.stride(1),
                C.stride(0), C.stride(1),
            )
        else:
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
            _matmul_generic_bt_kernel[grid](
                A_c, B_c, C,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                B_c.stride(0), B_c.stride(1),
                C.stride(0), C.stride(1),
            )

        return C


def run(*args):
    return ModelNew()(*args)
