import torch
import triton
import triton.language as tl

@triton.jit
def matmul_at_bT_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: one program per C tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and BT tiles
        A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)   # A is (M, K)
        BT_ptrs = BT + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk) # BT is (K, N) since B.T

        # Masks for boundaries
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        BT_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)

        # Load tiles and cast to float32 for accumulation
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        BT_tile = tl.load(BT_ptrs, mask=BT_mask, other=0.0).to(tl.float32)  # (BLOCK_K, BLOCK_N)

        # Accumulate using matmul on tiles
        acc += tl.dot(A_tile, BT_tile)

    # Write back to C (cast to float16 to match input dtype)
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA for Triton
        assert A.is_cuda and B.is_cuda, "Triton kernels require CUDA tensors"

        # Shapes
        M, K = A.shape
        N, K_b = B.shape
        assert K == K_b, f"Incompatible shapes: A is (M, K)={A.shape}, B is (N, K_b)={B.shape}"

        # Prepare B^T contiguous (B is (N, K) -> B.T is (K, N))
        BT = B.transpose(0, 1).contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Choose tile sizes: balance resource use and throughput
        # Using 128x64x32 ensures each program handles ~128 KB of shared data, which is generally safe.
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid dimensions
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = BT.stride(0)  # original B's row dim (N)
        stride_bk = BT.stride(1)  # original B's col dim (K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch kernel
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=4,
        )

        return C


def run(*args):
    return ModelNew()(*args)
