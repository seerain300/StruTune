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
    # 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Masks for edges
        mask_m = m_offsets < M
        mask_n = n_offsets < N
        mask_k = k_offsets < K

        # Pointers for A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Pointers for BT tile: BT is [K, N], we load BT[k, n] = B[n, k]
        BT_ptrs = BT + k_offsets[:, None] * stride_bTk + n_offsets[None, :] * stride_bTn

        # Load tiles with masks
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        bt = tl.load(BT_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate (fp32 for stability)
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Store results to C with mask
    C_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs: A [M, K], B [N, K]
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"B's second dim must match A's second dim: got K={K} vs K2={K2}"
        assert A.device.type == "cuda" and B.device.type == "cuda", "Inputs must be on CUDA device"

        # Explicitly create BT as contiguous transposed for predictable strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N], strides (K, 1)

        # Output in float32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes: chosen for robustness; could be tuned further after correctness
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

        # Cast to original dtype to match torch.matmul(A, B.T) behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
