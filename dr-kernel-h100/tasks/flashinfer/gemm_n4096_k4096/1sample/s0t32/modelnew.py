import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # This kernel assumes M == 1. Each program computes a BLOCK_N-wide slice of C for row 0.
    # We loop over K in chunks of BLOCK_K, compute outer products, and accumulate into C.
    # Addresses:
    #   A[0, k] -> A_ptr + 0*stride_am + k*stride_ak
    #   BT[k, n] -> BT_ptr + k*stride_bTk + n*stride_bTn
    #   C[0, n] -> C_ptr + 0*stride_cm + n*stride_cn

    # Column range for this program
    pid_n = tl.program_id(axis=0)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Load A[0, k] vector
        a_vec = tl.load(
            A + 0 * stride_am + k_range * stride_ak,
            mask=k_range < K,
            other=0.0,
        )  # shape: [BLOCK_K]

        # Load BT[k, n] tile
        # BT is [K, N], so stride_bTk is stride for k, stride_bTn is stride for n
        bt_tile = tl.load(
            BT + k_range[:, None] * stride_bTk + n_range[None, :] * stride_bTn,
            mask=(k_range[:, None] < K) & (n_range[None, :] < N),
            other=0.0,
        )  # shape: [BLOCK_K, BLOCK_N]

        # Accumulate outer products: acc[n] += sum_k a_vec[k] * bt_tile[k, n]
        # Do accumulation in fp32
        # Cast to fp32 for numerical stability
        a_vec = a_vec.to(tl.float32)
        bt_tile = bt_tile.to(tl.float32)

        # acc += sum over k of a_vec[k] * bt_tile[k, :]
        # Use tl.sum reduction over axis=0
        acc += tl.sum(a_vec[:, None] * bt_tile, axis=0)

    # Store results to C[0, n]
    tl.store(C + 0 * stride_cm + n_range * stride_cn, acc, mask=n_range < N)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D-tiled matmul: C[m, n] = sum_k A[m, k] * BT[k, n]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Load A[m, k] tile
        a_tile = tl.load(
            A + m_range[:, None] * stride_am + k_range[None, :] * stride_ak,
            mask=(m_range[:, None] < M) & (k_range[None, :] < K),
            other=0.0,
        )
        # Load BT[k, n] tile
        bt_tile = tl.load(
            BT + k_range[:, None] * stride_bTk + n_range[None, :] * stride_bTn,
            mask=(k_range[:, None] < K) & (n_range[None, :] < N),
            other=0.0,
        )

        # Cast to fp32 for accumulation
        a_tile = a_tile.to(tl.float32)
        bt_tile = bt_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(a_tile, bt_tile)

    # Store
    tl.store(
        C + m_range[:, None] * stride_cm + n_range[None, :] * stride_cn,
        acc,
        mask=(m_range[:, None] < M) & (n_range[None, :] < N),
    )


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [N, K]
        # Compute C = A @ B.T with shape [M, N]
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        assert A.dtype in (torch.float16, torch.float32, torch.bfloat16), "A must be a floating dtype"
        assert B.dtype in (torch.float16, torch.float32, torch.bfloat16), "B must be a floating dtype"

        M, K = A.shape
        N = B.shape[0]  # B is [N, K]

        # Explicitly create BT for correct strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Fast path for M == 1: row-wise kernel parallelizing along N
        if M == 1:
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
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
            # Generic 2D GEMM fallback
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast back to original dtype to match torch.matmul behavior
        return C.to(A.dtype)