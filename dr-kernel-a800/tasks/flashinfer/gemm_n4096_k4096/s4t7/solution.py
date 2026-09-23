import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_rowwise_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: [M, K]
    stride_bk, stride_bn,   # strides for B: [K, N]
    stride_cm, stride_cn,   # strides for C: [M, N]
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row m and a block of columns n
    m = tl.program_id(0)

    # Column offsets for this tile
    n_offsets = tl.arange(0, BLOCK_N)

    # FP32 accumulator for this row's columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_N):
        # We want to accumulate contributions for all n in n_offsets
        # For each k in the chunk, compute contribution A[m, k] * B[n, k]
        # Loop over k in small steps, but we can use vectorized indexing by iterating k0 + arange(BLOCK_N)
        # However, to keep it simple and robust, iterate over k one by one and accumulate.
        for k_step in range(0, BLOCK_N):
            k = k0 + k_step
            # Masks for scalar row and scalar k
            a_mask = (m < M) & (k < K)
            # Load A[m, k] as scalar
            a_val = tl.load(A_ptr + m * stride_am + k * stride_ak, mask=a_mask, other=0.0).to(tl.float32)

            # Load B[n, k] as vector across columns n
            b_ptrs = B_ptr + n_offsets * stride_bn + k * stride_bk
            b_mask = (n_offsets < N) & (k < K)
            b_vec = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

            # Outer accumulation for this k: acc[n] += a_val * b_vec[n]
            acc += a_val * b_vec

    # Store results for this row
    c_ptrs = C_ptr + m * stride_cm + n_offsets * stride_cn
    c_mask = (m < M) & (n_offsets < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    # Ensure CUDA tensors and contiguous memory
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Incompatible shapes for matmul: A is (*, {K}), B is ({K2}, *)")

    # Output tensor (FP16 like inputs), contiguous
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides for A [M, K], B [K, N], C [M, N]
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Grid: one program per row m
    grid = (M,)

    # Choose a column tile size; 128 works well and balances vectorization
    BLOCK_N = 128

    matmul_transpose_rowwise_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Replace the original run with Triton-based matmul(A, B.T)
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure tensors are on CUDA for Triton
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
