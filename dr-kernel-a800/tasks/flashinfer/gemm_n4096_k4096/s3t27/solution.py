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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D tiling over output C of shape (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Output pointers for the current tile
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # BT tile: BT has shape (K, N) since BT = B.T
        BT_ptrs = BT_ptr + (offs_k[:, None] * stride_bt + offs_n[None, :] * stride_bn)

        # Masks for boundary loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; cast to float32 for accumulation
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        # Use tl.dot for efficient matrix multiply on tiles
        acc += tl.dot(a, bt)

    # Store result (cast back to output dtype, float16 in this task)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
        # Shapes: A: (M, K), B: (N, K) -> BT: (K, N)
        M, K = A.shape
        N = B.shape[0]
        # Prepare B_T = B.T contiguous for efficient access
        BT = B.T.contiguous()

        # Output tensor (float16, matching input dtype)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bt, stride_bn = BT.stride(0), BT.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Tuned tile sizes: larger tiles + tl.dot should improve performance
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel with more warps and stages to improve throughput
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bt, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        return C


def run(*args):
    return ModelNew()(*args)
