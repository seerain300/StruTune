import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    A_stride0, A_stride1,
    BT_stride0, BT_stride1,
    C_stride0, C_stride1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: pid_m over M tiles, pid_n over N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers to current tiles
        A_tile_ptrs = A_ptr + m_offsets[:, None] * A_stride0 + k_offsets[None, :] * A_stride1
        BT_tile_ptrs = BT_ptr + k_offsets[:, None] * BT_stride0 + n_offsets[None, :] * BT_stride1

        # Masks for boundaries
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (masked), cast to fp32 for accumulation
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)
        BT_tile = tl.load(BT_tile_ptrs, mask=bt_mask, other=0.0)
        # Ensure fp32 accumulation
        A_tile = A_tile.to(tl.float32)
        BT_tile = BT_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Store back to output (cast to fp16 as needed)
    C_tile_ptrs = C_ptr + m_offsets[:, None] * C_stride0 + n_offsets[None, :] * C_stride1
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=out_mask)  # Triton will handle dtype conversion if C is fp16


def _choose_tiling(M, N, K):
    # Heuristic selection of tile sizes and scheduling
    # Small M: smaller BLOCK_M to reduce idle threads
    if M <= 32:
        BLOCK_M = 16
    elif M <= 128:
        BLOCK_M = 32
    else:
        BLOCK_M = 64

    # Large N: larger BLOCK_N reduces grid size along N
    if N <= 128:
        BLOCK_N = 64
    elif N <= 512:
        BLOCK_N = 128
    else:
        BLOCK_N = 256

    # K chunk size: use 16 for very small K, else 32
    BLOCK_K = 16 if K <= 16 else 32

    # Warps and stages: increase for larger tiles
    tile_area = BLOCK_M * BLOCK_N * BLOCK_K
    if tile_area <= 32 * 64 * 32:
        num_warps = 4
        num_stages = 3
    elif tile_area <= 64 * 128 * 32:
        num_warps = 8
        num_stages = 3
    else:
        num_warps = 8
        num_stages = 4

    return BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Preconditions: CUDA tensors, float16, 2D
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16."
        assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D."

        M, K = A.shape
        N, KB = B.shape
        assert KB == K, "B must have shape (N, K) matching A's K dimension."

        # Compute B.T contiguous for simple stride math
        BT = B.transpose(0, 1).contiguous()  # BT: (K, N)
        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Choose tiling based on shape
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _choose_tiling(M, N, K)

        # Grid over tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return C


def run(*args):
    return ModelNew()(*args)
