import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel_2d(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: (M, K)
    stride_bk, stride_bn,        # BT strides: (K, N)
    stride_cm, stride_cn,        # C strides: (M, N)
    BLOCK_M: tl.constexpr,       # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    # Note: K is runtime, so use Python range; masks ensure safety for partial chunks.
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A and BT tiles
        # A_tile: shape (BLOCK_M, BLOCK_K), BT_tile: shape (BLOCK_K, BLOCK_N)
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        BT_tile_ptrs = BT_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles with masks
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)
        BT_tile = tl.load(BT_tile_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(A_tile.to(tl.float32), BT_tile.to(tl.float32))

    # Write back to C (M, N), cast to output dtype (float16 expected)
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (M, K), B: (N, K)
        # Compute C = A @ B.T, result shape (M, N)
        # Ensure tensors are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "ModelNew requires CUDA tensors."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N = B.shape[0]  # B is (N, K)

        # Prepare BT = B.T contiguous
        BT = B.T.contiguous()  # BT is (K, N), float16

        # Allocate output C (M, N) in float16 to match original behavior
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides for A, BT, C
        stride_am = A.stride(0)  # typically K
        stride_ak = A.stride(1)  # typically 1
        stride_bk = BT.stride(0)  # typically N
        stride_bn = BT.stride(1)  # typically 1
        stride_cm = C.stride(0)   # typically N
        stride_cn = C.stride(1)   # typically 1

        # Choose tile sizes and launch grid
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel_2d[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        return C


def run(*args):
    return ModelNew()(*args)
