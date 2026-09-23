import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_i, BT_stride_k,  # BT is (K, N) after B.T
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets within the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Boundary masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks (runtime loop; avoid tl.static_range)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: shape [BLOCK_M, BLOCK_K], A is (M, K)
        A_tile_ptrs = A_ptr + (offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k)
        # BT tile: shape [BLOCK_K, BLOCK_N], BT is (K, N) after B.T
        BT_tile_ptrs = BT_ptr + (offs_k[:, None] * BT_stride_i + offs_n[None, :] * BT_stride_k)

        # Load with masks
        a = tl.load(A_tile_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        bt = tl.load(BT_tile_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Cast to float32 for accumulation
        a = a.to(tl.float32)
        bt = bt.to(tl.float32)

        # Accumulate using dot; Triton performs FMA on the tile
        acc += tl.dot(a, bt)

    # Store result to C (M, N), casting from float32 to float16
    C_tile_ptrs = C_ptr + (offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_tile_ptrs, acc.to(tl.float16), mask=mask)


def triton_matmul_at_bT(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B.T using Triton. A: (M, K), B: (N, K) -> C: (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton kernel."
    # Ensure contiguity; keep dtype as float16 (matching original)
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    N, Kb = B.shape
    assert Kb == K, f"B's second dimension must be K, got N={N}, K={K}, Kb={Kb}"

    # Compute B.T as contiguous tensor of shape (K, N)
    BT = B.transpose(0, 1).contiguous()

    # Allocate output in float16 (same as input dtype)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Tiling parameters tuned for performance without exceeding shared memory
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Grid: one program per tile
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


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on CUDA for Triton
        if not A.is_cuda or not B.is_cuda:
            A = A.cuda(non_blocking=True)
            B = B.cuda(non_blocking=True)
        return triton_matmul_at_bT(A, B)


def run(*args):
    return ModelNew()(*args)
