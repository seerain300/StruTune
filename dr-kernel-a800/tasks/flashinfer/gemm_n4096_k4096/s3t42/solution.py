import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A, BT, C,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_k, BT_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A + (offs_m[:, None] * A_stride_m) + (offs_k[None, :] * A_stride_k)
        # Pointers for BT tile: (BLOCK_K, BLOCK_N), BT shape (K, N)
        bt_ptrs = BT + (offs_k[:, None] * BT_stride_k) + (offs_n[None, :] * BT_stride_n)

        # Masks
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load with zero fill for masked lanes
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        BT_tile = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(A_tile.to(tl.float32), BT_tile.to(tl.float32))

    # Store back to C, masked
    c_ptrs = C + (offs_m[:, None] * C_stride_m) + (offs_n[None, :] * C_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        # Compute B.T as (K, N)
        BT = B.transpose(0, 1).contiguous()
        M, K = A.shape
        N = BT.shape[0]  # originally B.shape[1] after transpose

        # Output tensor (float16 to match inputs)
        out = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tile sizes
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, BT, out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )
        return out


def run(*args):
    return ModelNew()(*args)
