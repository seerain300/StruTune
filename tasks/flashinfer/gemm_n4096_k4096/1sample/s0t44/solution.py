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
    # 2D program id: tiles along M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A[m, k] and BT[k, n]
        # A is [M, K]: pointer = A + m * stride_am + k * stride_ak
        A_ptrs = A + (m[:, None] * stride_am + k[None, :] * stride_ak)
        # BT is [K, N]: pointer = BT + k * stride_bTk + n * stride_bTn
        BT_ptrs = BT + (k[:, None] * stride_bTk + n[None, :] * stride_bTn)

        # Masks for edges
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        bt_mask = (k[:, None] < K) & (n[None, :] < N)

        # Load tiles
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate (cast to fp32 for stability)
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Store result to C
    C_ptrs = C + (m[:, None] * stride_cm + n[None, :] * stride_cn)
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T
        assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, "B's last dim must equal A's last dim (K)"
        # Explicit transpose and make contiguous for predictable strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]
        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes: balanced defaults for modern GPUs
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
