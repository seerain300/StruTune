import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_n, BT_stride_k,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program id for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    # Using Python range loop to avoid Triton static_range issues with runtime K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers to A tile: shape [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + (offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k)
        # Masks for A
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

        # Pointers to BT tile: BT has shape [N, K], we want C_tile = A @ BT -> BT[k, n] layout
        BT_tile_ptrs = BT_ptr + (offs_n[None, :] * BT_stride_n + offs_k[:, None] * BT_stride_k)
        # Masks for BT
        BT_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)

        # Load tiles, cast to fp32 for accumulation
        A_tile = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)
        BT_tile = tl.load(BT_tile_ptrs, mask=BT_mask, other=0.0)

        # Cast to fp32 for better numeric stability
        A_tile = A_tile.to(tl.float32)
        BT_tile = BT_tile.to(tl.float32)

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(A_tile, BT_tile)

    # Write back to C
    # C has shape [M, N], so C_ptrs = C_ptr + m*stride_m + n*stride_n
    C_ptrs = C_ptr + (offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store as fp32; Triton will handle conversion to the destination tensor dtype
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A @ B.T using Triton kernels.
        - A: [M, K], float16 or float32
        - B: [N, K], float16 or float32
        Returns: C: [M, N]
        """
        # Ensure tensors are on CUDA; Triton requires CUDA device
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, "B's second dimension must match A's second dimension."

        # Make B.T contiguous for simpler indexing in Triton
        # Note: This is a pure PyTorch data movement (no computation), allowed as it's not host-side matmul.
        BT = B.transpose(0, 1).contiguous()

        # Output tensor (same dtype as A for consistency with original get_inputs)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Choose tile sizes; balance performance and shared memory usage
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64  # Larger K-chunk reduces loop iterations

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )
        return C


def run(*args):
    return ModelNew()(*args)
