import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: each program handles a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load BT tile: BT is [K, N], here we load rows k_offsets and columns n_offsets -> shape [BLOCK_K, BLOCK_N]
        bt_ptrs = BT + k_offsets[:, None] * stride_bTk + n_offsets[None, :] * stride_bTn
        bt = tl.load(bt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N)
        acc += tl.dot(a, bt)

    # Store the result tile
    c_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K, "Inner dimensions must match: B is [N, K] with K == A.shape[1]"

        # Create BT as a contiguous transposed view for predictable strides: BT is [K, N]
        BT = B.transpose(0, 1).contiguous()

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes tuned to be resource-friendly
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_generic_at_bt_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype to match torch.matmul(A, B.T) behavior
        return C.to(A.dtype)