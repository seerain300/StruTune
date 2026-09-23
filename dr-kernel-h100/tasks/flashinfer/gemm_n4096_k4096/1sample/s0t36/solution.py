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
    # Program ids for 2D tiling
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[m, k]
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # A is float16 by default in inputs

        # Pointers for BT[k, n] which is B[n, k] logically
        bt_ptrs = BT + (offs_k[:, None] * stride_bTk + offs_n[None, :] * stride_bTn)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(a, bt)

    # Write back results with masks
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [N, K]
        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K, "B must have shape [N, K] with the same K as A"

        # Explicitly create B_T as a contiguous tensor for predictable strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes tuned for general performance; masks handle edges
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

        # Cast to original dtype to match torch.matmul(A, B.T) behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
