import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel_2d(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,      # A strides: [M, K]
    stride_bk, stride_bn,      # BT strides: [K, N]
    stride_cm, stride_cn,      # C strides: [M, N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of output C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks (constexpr/static range)
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load BT tile: (BLOCK_K, BLOCK_N)
        BT_ptrs = BT_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        BT_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(BT_ptrs, mask=BT_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result to C
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton
        assert A.is_cuda and B.is_cuda, "Triton kernel requires CUDA tensors."

        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's last dim {Kb} must match A's last dim {K}"

        # Prepare B^T as (K, N) contiguous
        BT = B.T.contiguous()

        # Allocate output tensor (float16 like inputs)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tile sizes: tuned to improve parallelism for large N and small M
        BLOCK_M = 64
        BLOCK_N = 256
        BLOCK_K = 32

        # Grid over tiles of C
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel_2d[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        return C


def run(*args):
    return ModelNew()(*args)
