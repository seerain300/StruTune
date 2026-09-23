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
    # 2D tiling over output (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointers for A, BT, and C tiles
    a_ptrs = A + rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    bt_ptrs = BT + tl.arange(0, BLOCK_K)[:, None] * stride_bTk + rn[None, :] * stride_bTn
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs + k * stride_ak, mask=(rm[:, None] < M), other=0.0)
        bt = tl.load(bt_ptrs + k * stride_bTk, mask=(rn[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, bt)

    # Store result
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs
        assert A.dim() == 2, "A must be 2D [M, K]"
        assert B.dim() == 2, "B must be 2D [N, K]"
        M, K = A.shape
        N = B.shape[0]
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "A must be floating type"
        assert B.dtype in (torch.float16, torch.bfloat16, torch.float32), "B must be floating type"

        # Explicitly form B_T with correct shape and strides; using contiguous for predictable access
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output buffer in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes; masks handle edge tiles for arbitrary M, N, K
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