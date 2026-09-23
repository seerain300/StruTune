import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_at_bT_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bt, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Masks for boundaries
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_tile = tl.load(
            A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=a_mask,
            other=0.0
        )
        # Load BT tile: BT has shape (K, N), we load BT[k, n]
        BT_tile = tl.load(
            BT + k_offsets[:, None] * stride_bt + n_offsets[None, :] * stride_bk,
            mask=b_mask,
            other=0.0
        )

        # Cast to fp32 for accumulation
        A_tile = A_tile.to(tl.float32)
        BT_tile = BT_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Store results to C (mask boundaries)
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(
        C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=c_mask
    )


@triton.jit
def _row_matvec_bT_kernel(
    A_row, BT, C_row,
    K, N,
    stride_ar, stride_ak,
    stride_bt, stride_bk,
    stride_cr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program handles a tile of columns (BLOCK_N) for the single output row.
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for this row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A_row[k]
        a_vals = tl.load(
            A_row + k_offsets * stride_ak,
            mask=k_offsets < K,
            other=0.0
        ).to(tl.float32)  # (BLOCK_K,)

        # Load BT[k, n_offsets] tile
        b_tile = tl.load(
            BT + k_offsets[:, None] * stride_bt + n_offsets[None, :] * stride_bk,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0
        ).to(tl.float32)  # (BLOCK_K, BLOCK_N)

        # Accumulate contributions: acc += sum_k a_vals[k] * b_tile[k, :]
        acc += tl.sum(b_tile * a_vals[:, None], axis=0)

    # Store results to C_row
    c_mask = n_offsets < N
    tl.store(C_row + n_offsets * stride_cr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B.T using Triton kernels.
        A: (M, K), B: (N, K), output C: (M, N)
        """
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        M, K = A.shape
        N = B.shape[0]

        # Prepare BT = B.T as contiguous (K, N) for coalesced access
        BT = B.t().contiguous()

        # Output tensor
        out = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Specialized fast path for M == 1
        if M == 1:
            BLOCK_N = 128
            BLOCK_K = 128
            grid = (triton.cdiv(N, BLOCK_N),)
            A_row = A[0]  # shape (K,)
            C_row = out[0]  # shape (N,)
            _row_matvec_bT_kernel[grid](
                A_row, BT, C_row,
                K, N,
                A_row.stride(0), A_row.stride(1),
                BT.stride(0), BT.stride(1),
                C_row.stride(0),
                BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                num_warps=4, num_stages=3,
            )
            return out

        # General 2D kernel for M > 1
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_at_bT_kernel[grid](
            A, BT, out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out


def run(*args):
    return ModelNew()(*args)
