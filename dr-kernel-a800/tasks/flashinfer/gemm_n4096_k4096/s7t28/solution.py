import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K] with strides (stride_am, stride_ak)
# B: [K, N] with strides (stride_bk, stride_bn); B.T is treated as [N, K] with strides (B.stride(1), B.stride(0))
# C: [M, N], float16
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]; B_T[n, k] uses (B.stride(1), B.stride(0))
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = pid_k * BLOCK_K

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks
    for kk in range(0, BLOCK_K):
        k = k0 + kk
        # Load A_sub = A[offs_m, k] -> shape [BLOCK_M, 1]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + k * stride_ak)
        A_mask = (offs_m[:, None] < M) & (k < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, 1], fp16

        # Load B_sub_T = B_T[offs_n, k] -> shape [BLOCK_N, 1], where B_T[n, k] = B[n, k]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + k * stride_bk)  # B.stride(1) for n, B.stride(0) for k
        B_mask = (offs_n[:, None] < N) & (k < K)
        B_sub_T = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_N, 1], fp16

        # Accumulate: acc += A_sub [BLOCK_M, 1] @ tl.trans(B_sub_T [BLOCK_N, 1])
        # Note: tl.dot supports [BLOCK_M, 1] and [1, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub_T.to(tl.float32)))

    # Store result to C
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store as float16
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N]. Returns C: [M, N], float16.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton kernel."
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16."

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]."

    # Ensure contiguous for simple stride handling
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Choose tile sizes; these are safe defaults for fp16 GEMM
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
