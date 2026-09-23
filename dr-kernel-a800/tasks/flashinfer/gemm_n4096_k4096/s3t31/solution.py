import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiles along M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A is [M, K], BT is [K, N]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        bt_ptrs = BT_ptr + offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Cast to fp32 for accumulation
        a = a.to(tl.float32)
        bt = bt.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store result to C [M, N]; cast to original dtype (fp16 in the provided inputs)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"

        # Shapes: A [M, K], B [N, K]
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's second dimension must equal K of A, got K={K}, Kb={Kb}"

        # Make B^T contiguous for coalesced loads
        BT = B.T.contiguous()

        # Allocate output tensor (same dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_btk, stride_btn = BT.stride(0), BT.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Tiling parameters (conservative for robust correctness)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_btk, stride_btn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C


def run(*args):
    return ModelNew()(*args)
