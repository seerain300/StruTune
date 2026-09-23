import torch
import triton
import triton.language as tl


# Triton kernel: computes C = A @ B.T
# A: [M, K], B: [K, N] (PyTorch uses B.T = [N, K]); we access B[n, k] for B^T[k, n] = B[n, k].
# Output C: [M, N], float16 (to match original get_inputs).
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid: tiles over M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Output tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Multiply-accumulate: acc += A_sub @ B_sub^T
        # A_sub: [BM, BK], B_sub: [BN, BK] -> transpose B_sub to [BK, BN]
        acc += tl.dot(A_sub, tl.trans(B_sub))

    # Write back to C in fp16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Host-side wrapper: compute C = A @ B.T using Triton.
    A: [M, K], B: [K, N], both float16, on CUDA.
    Returns C: [M, N], float16.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton."
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16."
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]."

    # Ensure contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides (in elements)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tile sizes (tuned for general performance; adjust if needed)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # Grid must cover all dimensions; use lambda to compute grid from meta-parameters
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(K, meta['BLOCK_K']),
    )

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
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
