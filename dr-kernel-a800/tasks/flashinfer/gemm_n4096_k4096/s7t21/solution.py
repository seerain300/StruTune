import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transB_simple_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (scalar k to ensure correctness across all shapes)
    for k in range(0, K):
        # Load A[m, k] for all m in tile: shape [BLOCK_M]
        A_row_ptrs = A_ptr + (offs_m * stride_am + k * stride_ak)
        A_mask_row = (offs_m < M)
        A_vec = tl.load(A_row_ptrs, mask=A_mask_row, other=0.0)  # [BLOCK_M]

        # Load B_T[k, n] = B[n, k] for all n in tile: shape [BLOCK_N]
        B_col_ptrs = B_ptr + (offs_n * stride_bn + k * stride_bk)
        B_mask_col = (offs_n < N)
        B_vec = tl.load(B_col_ptrs, mask=B_mask_col, other=0.0)  # [BLOCK_N]

        # Outer product and accumulate: [BLOCK_M, 1] * [1, BLOCK_N]
        acc += A_vec[:, None].to(tl.float32) * B_vec[None, :].to(tl.float32)

    # Store result to C
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)  # acc is fp32; Triton will cast to C dtype if needed


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B.T where A is [M, K] and B is [K, N].
    Returns C with dtype float16, matching typical PyTorch behavior for the provided inputs.
    """
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure CUDA and contiguous tensors
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor: float16, matching original get_inputs()
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides in elements
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tile sizes for the simple kernel (can be tuned later)
    BLOCK_M = 64
    BLOCK_N = 128

    # 2D grid over M and N
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transB_simple_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,  # lower warps to keep register pressure reasonable in the simple kernel
        num_stages=2,
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
