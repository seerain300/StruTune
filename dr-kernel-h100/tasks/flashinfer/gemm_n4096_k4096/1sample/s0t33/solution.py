import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_matmul_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid is (1, N_tiles) when M == 1. Each program handles a block of columns.
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_cols = cols < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_range < K

        # Load A[0, k] as a vector
        # A is [M, K]; here M == 1, so address = A + 0*stride_am + k*stride_ak
        a_vec = tl.load(A + 0 * stride_am + k_range * stride_ak, mask=mask_k, other=0.0)

        # Load BT[k, cols] as a tile (shape [BLOCK_K, BLOCK_N])
        # BT is [K, N]; BT[k, cols] = B[cols, k]
        b_ptrs = BT + k_range[:, None] * stride_bTk + cols[None, :] * stride_bTn
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_cols[None, :], other=0.0)

        # Accumulate outer product: acc += sum_k a_vec[k] * b_tile[k, :]
        # Multiply [BLOCK_K] by [BLOCK_K, BLOCK_N] -> [BLOCK_N]
        acc += tl.sum(b_tile * a_vec[:, None], axis=0)

    # Store results to C[0, cols]
    tl.store(C + 0 * stride_cm + cols * stride_cn, acc, mask=mask_cols)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling across M and N with reduction over K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_m = m < M
        mask_n = n < N
        mask_k = k < K

        # Load A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + m[:, None] * stride_am + k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load BT[k, n] tile: shape [BLOCK_K, BLOCK_N]
        b_ptrs = BT + k[:, None] * stride_bTk + n[None, :] * stride_bTn
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Fused multiply-add: acc += A_tile @ B_tile
        acc += tl.dot(a_tile, b_tile)

    # Store results with masks
    c_ptrs = C + m[:, None] * stride_cm + n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure floating dtype
        assert A.dtype in (torch.float16, torch.float32), "A must be float16/float32"
        assert B.dtype in (torch.float16, torch.float32), "B must be float16/float32"

        # Compute C = A @ B.T
        M, K = A.shape
        N = B.shape[0]  # B is [N, K]

        # Explicitly create BT with correct shape and strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Allocate output in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Fast row-wise path
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_matmul_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic fallback for other M
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

        # Cast to original dtype to match torch.matmul behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
