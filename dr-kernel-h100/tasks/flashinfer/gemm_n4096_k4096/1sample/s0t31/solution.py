import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Pointers for A[m, k] and BT[k, n]
        a_ptrs = A + (offs_m[:, None] * stride_am) + ((k + offs_k)[None, :] * stride_ak)
        bt_ptrs = BT + ((k + offs_k)[:, None] * stride_bTk) + (offs_n[None, :] * stride_bTn)

        # Masks for boundary handling
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        bt_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles, cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, bt)

    # Write back to C
    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [N, K] => C = A @ B.T, shape [M, N]
        assert A.ndim == 2, "A must be 2D [M, K]"
        assert B.ndim == 2, "B must be 2D [N, K]"
        M, K = A.shape
        N, KB = B.shape
        assert KB == K, "B's second dimension must equal A's second dimension (K)"
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "A dtype must be a floating type"
        assert B.dtype in (torch.float16, torch.bfloat16, torch.float32), "B dtype must be a floating type"

        # Explicitly create B_T with correct shape and strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Allocate output in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes; masks handle arbitrary shapes
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_at_bt_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast to original dtype to match torch.matmul behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
