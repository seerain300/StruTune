import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel_fp16(
    A, BT, Out,
    M, N, K,
    A_stride0, A_stride1,
    BT_stride0, BT_stride1,
    Out_stride0, Out_stride1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D grid of tiles: (pid_m, pid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices this program will handle
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A sub-tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A + rows[:, None] * A_stride0 + k_range[None, :] * A_stride1
        # Mask for A loads
        a_mask = (rows[:, None] < M) & (k_range[None, :] < K)

        # Pointers for BT sub-tile: BT has shape (K, N), we want BT[k, cols] -> shape (BLOCK_K, BLOCK_N)
        bt_ptrs = BT + k_range[:, None] * BT_stride0 + cols[None, :] * BT_stride1
        # Mask for BT loads
        bt_mask = (k_range[:, None] < K) & (cols[None, :] < N)

        # Load tiles (fp16 inputs), cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, bt)

    # Write back to Out (fp16), masked by bounds
    out_ptrs = Out + rows[:, None] * Out_stride0 + cols[None, :] * Out_stride1
    out_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T
        # A: (M, K), B: (N, K), C: (M, N)
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, "B must have the same second dimension as A's second dimension"

        # Make B.T contiguous for simple stride handling in Triton
        BT = B.T.contiguous()  # shape (K, N), contiguous

        # Output tensor (float16, same as inputs)
        Out = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tile sizes: tuned for performance without excessive shared memory
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel_fp16[grid](
            A, BT, Out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            Out.stride(0), Out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return Out


def run(*args):
    return ModelNew()(*args)
