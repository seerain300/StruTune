import torch
import triton
import triton.language as tl

# 2D matmul kernel: computes C[M, N] = A[M, K] @ BT[K, N], where BT is B.T (contiguous KxN).
@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_k, BT_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and BT tiles
        A_tile_ptr = A_ptr + (offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k)
        BT_tile_ptr = BT_ptr + (offs_k[:, None] * BT_stride_k + offs_n[None, :] * BT_stride_n)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load as fp32
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0).to(tl.float32)
        BT_tile = tl.load(BT_tile_ptr, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Write back to C, cast to fp16 (original input dtype is fp16)
    C_out_ptr = C_ptr + (offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_out_ptr, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T
        # A: (M, K), B: (N, K), C: (M, N)
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, "Incompatible shapes for A and B"

        # If running on CPU or no CUDA, fallback to torch
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A, B.T)

        # For M == 1, use PyTorch matmul for speed and simplicity (avoid Triton kernel shape issues)
        if M == 1:
            return torch.matmul(A, B.T)

        # Prepare B^T contiguous as (K, N)
        BT = B.T.contiguous()

        # Output tensor (fp16 like input)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tile sizes: tuned for performance while keeping shared memory reasonable
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64  # larger K-chunk to reduce loop iterations

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
