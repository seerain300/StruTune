import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N] (original code uses B.T = [N, K]), we interpret B^T[k, n] = B[n, k].
# Output C: [M, N], float16 (to match get_inputs dtype).
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows (k), stride_bn over cols (n)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N; loop over K dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # fp32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_tile: A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_tile as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_sub @ B_sub^T
        # A_sub: [BLOCK_M, BLOCK_K], tl.trans(B_sub): [BLOCK_K, BLOCK_N]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub.to(tl.float32)))

    # Store result tile to C, cast to float16 (match original dtype)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N]. Returns C: [M, N], dtype=float16.
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D tensors"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]"
    # Ensure CUDA tensors and contiguous
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Choose block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_transB_kernel[grid](
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
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Use Triton to perform the matmul (A @ B.T)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
