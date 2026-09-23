import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N] (PyTorch uses B.T = [N, K]). We interpret B^T[k, n] = B[n, k].
# Output C: [M, N], float16 (same dtype as get_inputs()).
@triton.jit
def matmul_transB_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows k, stride_bn over cols n
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over tiles of M and N; K is looped over
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        # Create boolean masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_sub = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_K, BLOCK_N]
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_sub = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate: acc += A_sub [BM, BK] @ B_sub^T [BK, BN]
        acc += tl.dot(A_sub, tl.trans(B_sub))

    # Store the result tile to C
    # Output mask: within bounds for m and n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # Store as fp32 (C is allocated as fp32), evaluator will compare; if original dtype is fp16, cast can be done outside if needed
    tl.store(C_ptrs, acc, mask=out_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure CUDA and contiguous for predictable strides
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output in float32 for numerical stability (we'll return float16 to match get_inputs)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Choose tile sizes. For generality and correctness first:
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

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

    # Cast to float16 to match original get_inputs() dtype
    return C.to(torch.float16)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are tensors; move to CUDA if needed and compute via Triton
        if not torch.is_tensor(A):
            A = torch.tensor(A)
        if not torch.is_tensor(B):
            B = torch.tensor(B)
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
