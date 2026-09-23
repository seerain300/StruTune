import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N] (float16).
# We access B_T[k, n] = B[n, k], so load B with indices (n, k).
@triton.jit
def matmul_transB_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid: over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)

    # Initialize accumulator (fp32 for stability)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    # Note: we loop in Python/host to avoid 3D grid issues.
    # This kernel is launched repeatedly for each K chunk.
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: tl.dot([BLOCK_M, BLOCK_K], tl.trans([BLOCK_N, BLOCK_K])) -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub).to(tl.float32))

        k0 += BLOCK_K

    # Store result to C as float16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Cast to fp16 for output
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: [M, K], B: [K, N], both float16 on CUDA.
    Returns C: [M, N], float16, C = A @ B.T
    """
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure dtype float16 and contiguous on CUDA
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

    # Tile sizes: 64x64x64 is a good default for fp16; adjust if needed
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # 2D grid over M and N tiles
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Loop over K in chunks from host; we pass M, N, K and let the kernel handle one chunk per launch
    # In Triton, we cannot easily loop over K inside the kernel because of the while, but we can
    # launch the kernel multiple times by iterating k0 in Python. To avoid multiple kernel launches,
    # we instead restructure: call the kernel once and let it iterate over K internally by passing
    # K and BLOCK_K and using a while loop. Triton allows while loops.
    matmul_transB_kernel_2d[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
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
