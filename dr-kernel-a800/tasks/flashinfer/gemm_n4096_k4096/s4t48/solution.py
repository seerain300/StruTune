import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_simple_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K], typically after .contiguous(): stride_am=K, stride_ak=1
    stride_bk, stride_bn,   # B is [K, N], typically after .contiguous(): stride_bk=N, stride_bn=1
    stride_cm, stride_cn,   # C is [M, N], typically after .contiguous(): stride_cm=N, stride_cn=1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row/col indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create masks for boundary conditions
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension one by one to minimize risk of indexing mistakes
    for k in range(0, K):
        # Load A[m, k] as a vector of length BLOCK_M
        a_ptrs = A_ptr + m_offsets * stride_am + k * stride_ak
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)  # shape [BLOCK_M]

        # Load B_T[k, n] = B[n, k] as a vector of length BLOCK_N
        b_ptrs = B_ptr + n_offsets * stride_bn + k * stride_bk
        b = tl.load(b_ptrs, mask=mask_n, other=0.0)  # shape [BLOCK_N]

        # Expand to matrices and accumulate: a[:, None] * b[None, :]
        # This broadcasts a across N and b across M for the tile.
        acc += a[:, None] * b[None, :]

    # Store results back to C; Triton will cast to C's dtype if needed
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

    # Ensure contiguity for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise RuntimeError(f"Incompatible shapes: A is [M, K]={A.shape}, B is [Kb, N]={B.shape}")

    # Allocate output, same dtype as input A (FP16 in the given setup)
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Choose conservative tile sizes for robust correctness
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transpose_simple_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect A and B as inputs
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # The original code uses torch.matmul(A, B.T). We replace it here with Triton.
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
