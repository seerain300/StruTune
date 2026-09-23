import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A (shape [M, K]) after .contiguous()
    stride_bk, stride_bn,   # strides for B (shape [K, N]) after .contiguous()
    stride_cm, stride_cn,   # strides for C (shape [M, N]) after .contiguous()
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids: tile indices over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C (index into A)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C (index into B^T)

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k] with shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile: B_T[k, n] = B[n, k], so indexing uses k as rows and n as cols
        # B has shape [K, N] with strides (stride_bk, stride_bn)
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store results to C[m, n] as FP16 (match original example dtype)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A, B):
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], C: [M, N].
    All computation is done via Triton kernels. No torch.matmul on host.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."

    # Ensure contiguous for simple, coalesced indexing
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: A is {A.shape}, B is {B.shape} (expected K=={K})"

    # Allocate output tensor (FP16 to match example; accumulation in FP32 inside kernel)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides after .contiguous()
    stride_am, stride_ak = A.stride(0), A.stride(1)  # typically (K, 1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)  # typically (N, 1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)  # typically (N, 1)

    # Choose tile sizes; conservative and robust for varied dimensions
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transpose_kernel[grid](
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
    def forward(self, *args):
        # Triton-only implementation: compute A @ B.T via Triton kernels.
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure tensors are on CUDA device for Triton
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        # Run Triton kernel
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
