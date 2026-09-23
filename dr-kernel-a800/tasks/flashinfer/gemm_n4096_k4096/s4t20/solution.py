import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides after .contiguous(): typically stride_am=K, stride_ak=1
    stride_bk, stride_bn,   # B strides after .contiguous(): typically stride_bk=N, stride_bn=1
    stride_cm, stride_cn,   # C strides after .contiguous(): typically stride_cm=N, stride_cn=1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program handles a tile [BLOCK_M, BLOCK_N] of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: logically B_T[k, n] = B[n, k], shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C (FP16), with masks
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using a 2D-tiled Triton kernel.
    A: [M, K], B: [K, N], C: [M, N]
    """
    if not A.is_cuda or not B.is_cuda:
        # Fallback to PyTorch if tensors are not on CUDA
        return A @ B.T

    # Ensure contiguity and correct dtypes
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]")

    # Output tensor (FP16 to match original example; accumulation is FP32 in-kernel)
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides for contiguous tensors
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tiling parameters: tuned for large matrices
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transpose_tiled_kernel[grid](
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
    def forward(self, *args):
        # Expect two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        # Compute C = A @ B.T via Triton
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
