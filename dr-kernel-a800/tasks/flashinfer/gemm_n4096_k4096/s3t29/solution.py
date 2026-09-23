import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bt, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling
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

        # Pointers for A and B_T tiles
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        bt_ptrs = BT_ptr + offs_k[:, None] * stride_bt + offs_n[None, :] * stride_bn

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; promote to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Cast to fp32 for stable accumulation
        a = a.to(tl.float32)
        bt = bt.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Cast back to original dtype (float16)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous where needed
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, "B's second dimension must equal A's second dimension (K)"
        # Prepare B^T contiguous for coalesced access along K
        BT = B.T.contiguous()

        # Allocate output (match A's dtype)
        out = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bt = BT.stride(0)
        stride_bn = BT.stride(1)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        # Tile sizes and grid
        # Choose moderate tiles to balance performance and resource usage
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, BT, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bt, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,  # balanced for these tile sizes
            num_stages=4,  # pipelining along K
        )

        return out


def run(*args):
    return ModelNew()(*args)
