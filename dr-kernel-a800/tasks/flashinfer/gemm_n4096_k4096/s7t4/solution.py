import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N]
# We interpret B^T[k, n] = B[n, k] and load B[n, k] accordingly.
@triton.jit
def matmul_transB_kernel_3d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows (k), stride_bn over cols (n)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-chunk start for this program
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)

    # Masks for loads
    A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

    # Load A_sub: A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

    # Load B_sub as B^T: B^T[k, n] = B[n, k] -> B[offs_n, offs_k] -> [BLOCK_N, BLOCK_K]
    B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

    # Accumulate partial result: [BLOCK_M, BLOCK_K] dot [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
    acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub.to(tl.float32)))

    # Store results to C as float16 (match original dtype)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N], dtype float16.
    A and B must be on CUDA device and float16.
    """
    assert A.is_cuda and B.is_cuda, "A and B must be on CUDA for Triton execution"
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Incompatible shapes: A is [M, {K}], B is [{K_b}, N]"

    # Allocate output tensor (float16 to match original get_inputs())
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Tile sizes tuned for performance and robustness
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    # 3D grid over M, N, and K tiles to ensure full coverage
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel_3d[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
