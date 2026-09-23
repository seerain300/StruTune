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
    # Program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and BT tiles
        A_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        BT_ptrs = BT + offs_k[:, None] * stride_bTk + offs_n[None, :] * stride_bTn

        # Masks for edge tiles
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; use fp32 accumulation
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        BT_tile = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Write back to C
    C_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on the same device and have correct shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K, "Inner dimensions must match: B is [N, K] with K == A.shape[1]"

        # Create BT as a contiguous transposed view for predictable strides
        # Even if B is non-contiguous, .contiguous() produces a well-defined tensor
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes and launch grid
        BLOCK_M = 64
        BLOCK_N = 256
        BLOCK_K = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_generic_at_bt_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=4,
        )

        # Cast back to original dtype to match torch.matmul(A, B.T) behavior
        return C.to(A.dtype)