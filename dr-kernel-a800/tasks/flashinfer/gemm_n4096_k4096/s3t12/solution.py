import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_2d_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides: (M, K)
    stride_btk, stride_btn,  # BT strides: (K, N)
    stride_cm, stride_cn,    # C strides: (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks (runtime loop; Triton supports this for dynamic K)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: shape (BLOCK_K, BLOCK_N)
        bt_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()

        # BT = B.T as contiguous (K, N)
        BT = B.transpose(0, 1).contiguous()

        # Shapes
        M, K = A.shape
        K_B, N = BT.shape
        if K_B != K:
            raise RuntimeError(f"Incompatible shapes: A is (M={M}, K={K}), B.T should have first dim K, got BT shape {BT.shape}")

        # Output tensor (float16 like inputs)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_btk = BT.stride(0)
        stride_btn = BT.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Kernel launch configuration: conservative tiles to avoid shared memory issues
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_2d_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_btk, stride_btn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,  # balanced parallelism
            num_stages=2, # modest pipelining
        )
        return C


def run(*args):
    return ModelNew()(*args)
