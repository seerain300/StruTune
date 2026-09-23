import torch
import triton
import triton.language as tl

# Robust 2D GEMM: computes C = A @ BT, where BT = B.transpose(0, 1)
@triton.jit
def _matmul_a_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile index along M
    pid_n = tl.program_id(1)  # tile index along N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first tiles
    A_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    BT_ptrs = BT + offs_k[:, None] * stride_bTk + offs_n[None, :] * stride_bTn  # [BK, BN]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_mask_a = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        k_mask_bt = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=k_mask_a, other=0.0)  # [BM, BK]
        bt = tl.load(BT_ptrs, mask=k_mask_bt, other=0.0)  # [BK, BN]
        # Accumulate (acc is fp32)
        acc += tl.dot(a, bt)  # [BM, BN]
        # Advance pointers along K
        A_ptrs += BLOCK_K * stride_ak
        BT_ptrs += BLOCK_K * stride_bTk

    # Write back to C with masks for edge tiles
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N, K_b = B.shape
        assert K_b == K, f"B's last dim must match A's last dim, got K={K} vs K_b={K_b}"

        # Explicitly create BT = B.transpose(0, 1) and make it contiguous for predictable strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes chosen for good throughput on large matrices (e.g., 4096x4096)
        # These are robust and should work for arbitrary M, N, K.
        BLOCK_M = 128
        BLOCK_N = 256
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_a_bt_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8,  # higher parallelism per program for large tiles
            num_stages=3,  # pipeline depth
        )

        # Cast to original dtype to match torch.matmul(A, B.T) behavior
        return C.to(A.dtype)