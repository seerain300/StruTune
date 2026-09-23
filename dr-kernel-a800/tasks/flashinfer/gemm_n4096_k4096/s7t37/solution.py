import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# - A is [M, K], B is [K, N]
# - B_T[k, n] = B[n, k]
# - Output C is [M, N]
@triton.jit
def matmul_transB_kernel(
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

    # Compute output tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_sub @ B_sub.T -> [BLOCK_M, BLOCK_N]
        # B_sub is [BLOCK_N, BLOCK_K]; transpose to [BLOCK_K, BLOCK_N]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub.to(tl.float32)))

    # Store result as float16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. Assumes A: [M, K], B: [K, N].
    Returns C: [M, N] in float16.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device for Triton."
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Use float16 tensors."

    # Ensure contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Incompatible shapes: A is [M,{K}], B is [{K_b},N]"

    # Allocate output as float16 to match original behavior
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides in elements
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tile sizes: tuned for general performance; masks ensure correctness for any M,N,K
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    # 3D grid covering all dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=4,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors; if not provided, move to CUDA
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
