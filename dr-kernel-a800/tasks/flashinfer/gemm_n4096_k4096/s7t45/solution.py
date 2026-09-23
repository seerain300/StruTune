import torch
import triton
import triton.language as tl

# Triton kernel: computes C = A @ B.T
# A: [M, K], B: [K, N] in PyTorch, but we only need B^T[k, n] = B[n, k].
# We implement B^T by loading B[n, k].
@triton.jit
def matmul_transB_kernel_2d_kloop(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride over k (rows) and n (cols)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator (fp32 for robustness)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B tile corresponding to B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: tl.dot expects both inputs to be the same tile shape; we pass B_sub and transpose it on the fly.
        # B_sub is [BLOCK_N, BLOCK_K]; we want [BLOCK_K, BLOCK_N]. Use tl.trans(B_sub).
        acc += tl.dot(A_sub, tl.trans(B_sub))

        k0 += BLOCK_K

    # Store result to C in fp16 to match original dtype
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Cast to fp16 before store
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N]; output C: [M, N]
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is (*, {K}), B is ({Kb}, {N})"
    # Ensure dtypes and device
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"
    # Make contiguous for simple stride handling
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides in elements
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tiling parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # 2D grid over M and N tiles
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transB_kernel_2d_kloop[grid](
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
        # Ensure tensors are on CUDA for Triton execution; get_inputs() produces CPU tensors, so move to CUDA
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
