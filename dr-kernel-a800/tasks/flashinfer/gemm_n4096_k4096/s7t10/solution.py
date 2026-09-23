import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T
# A: [M, K], B: [K, N], output C: [M, N], float16
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # strides for A (row, col)
    stride_bk, stride_bn,       # strides for B (row k, col n) since B is [K, N]
    stride_cm, stride_cn,       # strides for C (row m, col n)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-chunk start and indices
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)

    # K-bound validity for this chunk
    mask_k = offs_k < K

    # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    A_mask = (offs_m[:, None] < M) & (mask_k[None, :])
    A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

    # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
    B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    B_mask = (offs_n[:, None] < N) & (mask_k[None, :])
    B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

    # Accumulate: A_sub [BM, BK] @ B_sub^T [BK, BN] -> acc [BM, BN]
    acc += tl.dot(A_sub, tl.trans(B_sub))

    # Store results to C as fp16 to match original dtype
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=store_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton.
    A: [M, K], B: [K, N], returns C: [M, N] float16.
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M,{K}], B is [{Kb},N]"
    # Ensure contiguous for simple stride handling and coalesced access
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output on CUDA (float16 to match original)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides in elements
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tile sizes tuned for robust performance across shapes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128

    # 3D grid covers M, N, and K fully
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

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
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
