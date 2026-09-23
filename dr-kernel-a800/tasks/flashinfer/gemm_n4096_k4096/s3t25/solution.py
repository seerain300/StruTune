import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_k, BT_stride_n,       # BT is a 2D tensor with shape (K, N)
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and BT tiles
        A_tile_ptr = A_ptr + (offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k)
        BT_tile_ptr = BT_ptr + (offs_k[:, None] * BT_stride_k + offs_n[None, :] * BT_stride_n)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (convert to float32 for accumulation)
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        BT_tile = tl.load(BT_tile_ptr, mask=bt_mask, other=0.0)

        # Cast to float32 for accumulation
        A_tile = A_tile.to(tl.float32)
        BT_tile = BT_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Store results back to C (cast to original dtype of A, which is float16 in this task)
    C_tile_ptr = C_ptr + (offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Note: We cast to tl.float16 before store; if you want to support other dtypes, adjust accordingly.
    tl.store(C_tile_ptr, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA; Triton requires CUDA
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton execution."
        # Compute B^T and make it contiguous
        BT = B.transpose(0, 1).contiguous()  # BT shape: (K, N)
        M = A.shape[0]
        K = A.shape[1]
        N = BT.shape[1]  # B has shape (N, K), so BT is (K, N)
        # Allocate output
        out = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Tile sizes: conservative to avoid shared memory limits
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
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
