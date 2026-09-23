import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create pointers for the first K tile
    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: shape (BLOCK_M, BLOCK_K)
        A_tile_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Pointers for BT tile: shape (BLOCK_K, BLOCK_N), BT is B transposed and contiguous: BT[k, n]
        BT_tile_ptrs = BT_ptr + k_offsets[:, None] * stride_btk + n_offsets[None, :] * stride_btn

        # Masks for bounds
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        BT_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles. Since A and BT could be fp16/bf16/fp32, cast to fp32 for accumulation.
        A_tile = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)
        BT_tile = tl.load(BT_tile_ptrs, mask=BT_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile.to(tl.float32), BT_tile.to(tl.float32))

    # Store results back to C. We store in original dtype of C (here, same as A dtype).
    # C has shape (M, N) with strides (stride_cm, stride_cn).
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A @ B.T, where A: [M, K], B: [N, K], returns C: [M, N].
        All computation is done in Triton; no torch.matmul is used on tensors in the forward path.
        """
        assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
        M, K = A.shape
        N, K_B = B.shape
        assert K == K_B, "B's second dimension must match A's second dimension (K)"
        # Ensure dtype consistency: compute in fp32 for accumulation, but return in original dtype of A
        # We'll allocate C with the same dtype as A to match PyTorch behavior.
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Make B_T contiguous for coalesced loads
        BT = B.T.contiguous()

        # Strides (in elements)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_btk, stride_btn = BT.stride(0), BT.stride(1)  # BT is (K, N)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Choose conservative tile sizes to maintain correctness across diverse shapes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_btk, stride_btn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C


def run(*args):
    return ModelNew()(*args)
