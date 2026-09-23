import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N] (PyTorch uses B.T = [N, K]). We interpret B^T[k, n] = B[n, k].
# Output C: [M, N], float16.
@triton.jit
def matmul_transB_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows k, stride_bn over cols n
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)  # fp16

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)  # fp16

        # Accumulate: acc += A_sub @ B_sub^T
        # A_sub: [BLOCK_M, BLOCK_K], tl.trans(B_sub): [BLOCK_K, BLOCK_N]
        acc += tl.dot(A_sub, tl.trans(B_sub))

        k0 += BLOCK_K

    # Store result to C (cast to fp16 to match output dtype)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"
    # Allocate output tensor (fp16 to match inputs)
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Tile sizes chosen for balanced performance and correctness
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    # 2D grid over M and N tiles
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transB_kernel_2d[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
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
