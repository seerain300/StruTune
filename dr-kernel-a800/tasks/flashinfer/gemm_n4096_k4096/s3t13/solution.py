import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: shape (BLOCK_M, BLOCK_K)
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_tile_ptrs, mask=A_mask, other=0.0).to(tl.float32)

        # Pointers for BT tile: BT has shape (K, N), we index as (offs_k, offs_n)
        BT_tile_ptrs = BT_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        BT_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(BT_tile_ptrs, mask=BT_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back to C (cast to output dtype if needed; here inputs are fp16 and output is fp16)
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        # Make sure tensors are contiguous and in the expected dtype
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape  # B is (N, K) originally, but here we pass B.T so we need (K, N)
        assert Kb == K, f"B must have shape (N, K) with K={K}, got {B.shape}"
        # Compute B.T contiguous
        BT = B.transpose(0, 1).contiguous()  # BT: (K, N)

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = BT.stride()
        stride_cm, stride_cn = C.stride()

        # Tile and launch configuration. Use larger tiles to improve throughput, but keep within shared memory limits.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8,  # more warps for better throughput
            num_stages=3,  # pipeline stages to overlap memory and compute
        )

        return C


def run(*args):
    return ModelNew()(*args)
