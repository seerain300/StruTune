import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small M cases: BLOCK_M=1 minimizes masked lanes
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512,  'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512,  'BLOCK_K': 128}, num_warps=8,  num_stages=3),
        # Slightly larger M
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 512,  'BLOCK_K': 64},  num_warps=16, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_1d_n_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Initialize accumulator [BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over rows of A (M dimension)
    # We keep BLOCK_M small to minimize masked work when M is tiny
    for m_idx in range(0, M):
        # Loop over K in chunks
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            # Load A[m_idx, k:k+BLOCK_K] -> [BLOCK_K, 1] then broadcast to [BLOCK_K, BLOCK_N]
            a_ptrs = A_ptr + m_idx * stride_am + offs_k * stride_ak  # shape [BLOCK_K]
            a = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K, 1]
            a = a[:, None]  # broadcast across N tile

            # Load B_T[k:k+BLOCK_K, n:n+BLOCK_N] -> [BLOCK_K, BLOCK_N]
            b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

            # Accumulate
            acc += a @ b  # [BLOCK_M, BLOCK_N] += [1, BLOCK_K] @ [BLOCK_K, BLOCK_N] when BLOCK_M=1

    # Store result
    c_ptrs = C_ptr + (tl.arange(0, BLOCK_M)[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # Broadcast masks across M dimension for store
    store_mask = (tl.arange(0, BLOCK_M)[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args

        # Device check: must be CUDA for Triton
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew requires CUDA tensors. Please move inputs to CUDA.")

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A is [M, K], B is arbitrary; we will treat B.T as [K, N] by indexing with swapped strides
        if A.dim() != 2:
            raise ValueError(f"A must be 2D [M, K], got shape {tuple(A.shape)}.")
        M, K = A.shape

        # Output tensor C [M, N] (N inferred from B's second dimension as K for B.T)
        # We don't know N from B shape directly here, because B.T requires B to have at least 1 dim.
        # The reference run(A, B) uses B.T unconditionally; since we cannot infer N from B in general,
        # we will infer N from B.T's shape via B's strides: if B is 1D, N = B.shape[0]; if B is ND, we need to
        # materialize B.T. To avoid falling back to torch (as per feedback), we instead materialize B.T with
        # .t().contiguous() on CUDA. This keeps Triton usage while matching reference semantics.
        # Note: This uses PyTorch only to form B.T, but does not compute the matmul in PyTorch; the kernel
        # still performs the actual matmul.
        BT = B.t().contiguous()
        N, K_b = BT.shape
        if K_b != K:
            raise ValueError(f"B.T's first dim (K of BT) must match A.shape[1] (K). Got K={K_b}, expected {K}.")

        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides for A, BT, C
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = BT.stride(0)  # corresponds to original B's second dim (N)
        stride_bn = BT.stride(1)  # corresponds to original B's first dim (K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: 1D over N tiles
        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_N']),)

        _matmul_1d_n_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
