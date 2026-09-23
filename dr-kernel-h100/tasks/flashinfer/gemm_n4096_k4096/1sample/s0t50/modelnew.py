import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_elementwise_reduce_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling: each program computes a [BLOCK_M, BLOCK_N] tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load BT tile: BT is [K, N]; we want BT[k, n] -> shape [BLOCK_K, BLOCK_N]
        BT_ptrs = BT + k_offsets[:, None] * stride_bTk + n_offsets[None, :] * stride_bTn
        bt = tl.load(BT_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Elementwise multiply and sum over K-chunk: broadcast to [BLOCK_M, BLOCK_N]
        # Multiply a[:, :, None] with bt[None, :, :] to get [BLOCK_M, BLOCK_K, BLOCK_N]
        prod = a[:, :, None] * bt[None, :, :]
        # Reduce over K-axis (axis=1), yielding [BLOCK_M, BLOCK_N]
        acc += tl.sum(prod, axis=1)

    # Store results with masks for edge tiles
    C_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [N, K]
        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K, "B must have shape [N, K] with the same K as A"

        # Create BT as a transposed view (no data movement)
        BT = B.transpose(0, 1)  # BT: [K, N]

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes: modest defaults to balance stability and performance
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_elementwise_reduce_kernel[grid](
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