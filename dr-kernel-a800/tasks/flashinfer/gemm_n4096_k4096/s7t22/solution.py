import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T, where
# A: [M, K], B: [K, N], output C: [M, N]
# We interpret B^T[k, n] = B[n, k]. So we load B[offs_n, offs_k] per K tile.
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows (k), stride_bn over cols (n)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)

        # Load A_sub = A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_sub = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_sub corresponding to B^T: B^T[k, n] = B[n, k]
        # We load B[offs_n, offs_k] -> [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_sub = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate: acc += A_sub @ transposed(B_sub)
        # A_sub: [BM, BK], B_sub: [BN, BK] -> transposed(B_sub): [BK, BN]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub.to(tl.float32)))

    # Store result as float16 to match torch.matmul default dtype in the original code
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N]
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure float16 and contiguous for predictable strides
    if A.dtype != torch.float16:
        A = A.to(torch.float16)
    if B.dtype != torch.float16:
        B = B.to(torch.float16)
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output tensor
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides in elements
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tiling parameters
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # 3D grid: cover all M, N tiles and all K chunks
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    # Launch kernel
    matmul_transB_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors; move to CUDA if needed
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
