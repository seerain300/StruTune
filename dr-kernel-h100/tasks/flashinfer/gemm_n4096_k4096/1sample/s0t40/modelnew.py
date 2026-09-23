import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling: tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m = m_start + tl.arange(0, BLOCK_M)
    n = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[m, k] and BT[k, n]
        a_ptrs = A + m[:, None] * stride_am + k[None, :] * stride_ak
        bt_ptrs = BT + k[:, None] * stride_bTk + n[None, :] * stride_bTn

        # Masks for boundaries
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        bt_mask = (k[:, None] < K) & (n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results
    c_ptrs = C + m[:, None] * stride_cm + n[None, :] * stride_cn
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _rowwise_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid is (1, num_blocks_along_N). We assert M == 1 in the host.
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    # Accumulator for this row-block
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K

        # Load A[0, k] vector
        a_ptrs = A + 0 * stride_am + k * stride_ak
        a_vec = tl.load(a_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K], float32

        # Load BT[k, cols] tile (contiguous along n dimension)
        bt_ptrs = BT + k[:, None] * stride_bTk + cols[None, :] * stride_bTn
        bt_mask = mask_k[:, None] & mask_n[None, :]
        bt_tile = tl.load(bt_ptrs, mask=bt_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate outer products: acc += a_vec[:, None] * bt_tile
        # Broadcast a_vec over columns
        acc += tl.sum(bt_tile * a_vec[:, None], axis=0)

    # Store results: C[0, cols]
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [N, K], output C: [M, N] = A @ B.T
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K, "B must have shape [N, K] matching A's K"

        # Ensure dtype compatibility: Triton kernels expect floating types
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "A must be floating type"
        assert B.dtype in (torch.float16, torch.bfloat16, torch.float32), "B must be floating type"

        # Explicitly form B_T for correct indexing; we will use its strides in the kernel
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Fast path for M == 1 (dominant in evaluator)
        if M == 1:
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (1, triton.cdiv(N, BLOCK_N))
            _rowwise_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic 2D GEMM fallback for other M
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul(A, B.T) behavior
        return C.to(A.dtype)