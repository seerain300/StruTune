import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T
# A: [M, K], B: [K, N]. B^T[k, n] = B[n, k].
# We tile over M and N and loop over K in chunks.
@triton.jit
def matmul_transB_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,      # strides for A: A is [M, K]
    stride_bk, stride_bn,      # strides for B: B is [K, N]
    stride_cm, stride_cn,      # strides for C: C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub corresponding to B^T: B^T[k, n] = B[n, k]
        # We load B[offs_n, offs_k] -> [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_sub @ B_sub.T
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub).to(tl.float32))

    # Store results to C[offs_m, offs_n]
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton, A: [M, K], B: [K, N], output: [M, N].
    """
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure CUDA tensors
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)

    # Make contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor (float16 to match provided get_inputs)
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides (in elements)
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tile sizes (tuned for robustness and performance)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # Grid over tiles; cover all dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_tiled_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
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
