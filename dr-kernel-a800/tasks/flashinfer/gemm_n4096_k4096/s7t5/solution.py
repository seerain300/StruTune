import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T
# A: [M, K], B: [K, N], output C: [M, N]
# B^T[k, n] = B[n, k] -> we load B[n, k] directly.
@triton.jit
def matmul_transB_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides (row-major: stride_am over M, stride_ak over K)
    stride_bk, stride_bn,   # B strides (B is [K, N]: stride_bk over rows k, stride_bn over cols n)
    stride_cm, stride_cn,   # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K] from A[offs_m, offs_k]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)  # load as fp16; acc is fp32

        # Load B tile as B^T: [BLOCK_K, BLOCK_N], with B^T[k, n] = B[n, k]
        # Access B[n, k] by B_ptr + n * stride_bn + k * stride_bk
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)  # load as fp16

        # Accumulate: acc += A_sub @ B_sub over K
        # A_sub: [BLOCK_M, BLOCK_K], B_sub: [BLOCK_K, BLOCK_N]
        acc += tl.dot(A_sub, B_sub)  # fp16 dot with fp32 acc

    # Store results to C in fp16 (C is allocated as fp16)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)  # acc is fp32; Triton will cast to fp16 on store


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)

        # Make inputs contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Allocate output as float16 to match original dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)  # over rows k
        stride_bn = B.stride(1)  # over cols n
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters (balanced for throughput and stability)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        # 2D grid over M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_transB_2d_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
