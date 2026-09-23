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
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for current tile of A: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # Pointers for current tile of B^T: shape [BLOCK_K, BLOCK_N]
        # BT has shape (K, N) and strides (stride_bt, stride_bn)
        bt_ptrs = BT_ptr + (offs_k[:, None] * stride_bt + offs_n[None, :] * stride_bn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)      # [BLOCK_M, BLOCK_K], fp16
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)   # [BLOCK_K, BLOCK_N], fp16

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Write back the tile to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store as fp16 to match input dtype
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: (M, K), B: (N, K), compute C = A @ B.T  -> C: (M, N)
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton kernel"
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"

        M, K = A.shape
        N = B.shape[0]  # B is (N, K)

        # Prepare B^T contiguous for Triton (shape (K, N))
        BT = B.T.contiguous()

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Tile sizes: conservative and performant
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C


def run(*args):
    return ModelNew()(*args)
