import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_tile_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: A is [M, K]
    stride_bk, stride_bn,   # strides for B: B is [K, N]
    stride_cm, stride_cn,   # strides for C: C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row/col indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this tile (FP32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # K offsets for this chunk
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[m, k] with shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B_T tile: B_T[k, n] = B[n, k], with shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # Since tl.dot requires 2D tiles, we use broadcasting and reduction:
        # acc += sum over K of a[:, kk] * b[kk, :]. We can achieve this by:
        for kk in range(BLOCK_K):
            acc += a[:, kk][:, None] * b[kk, :][None, :]

    # Store result to C with masks
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of C = A @ B.T
    A: [M, K], B: [K, N], C: [M, N]
    """
    # Ensure CUDA and contiguous
    if not A.is_cuda or not B.is_cuda:
        # If not CUDA, move to CUDA to use Triton kernels
        A = A.cuda(non_blocking=True)
        B = B.cuda(non_blocking=True)
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Incompatible shapes for A: {A.shape} and B: {B.shape}. Expected A[M,K], B[K,N].")

    # Output tensor, FP16 like inputs
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Choose conservative block sizes for robustness
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32  # small K chunk for safe masking and compilation robustness

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_transpose_tile_kernel[grid](
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
    def forward(self, *args):
        # Triton-only implementation of run(A, B) = A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
