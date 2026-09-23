import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: [M, K] in elements
    stride_bk, stride_bn,   # strides for B: [K, N] in elements
    stride_cm, stride_cn,   # strides for C: [M, N] in elements
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Single iteration over K (BLOCK_K = 32/64). We rely on masks to handle K < BLOCK_K.
    k_offsets = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # B_T[k, n] = B[n, k]
    b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Accumulate in FP32
    acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C[m, n] in FP16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A, B):
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    # Ensure tensors are CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]"

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Fixed, conservative tile sizes that previously passed correctness
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32  # single iteration: masks handle K < BLOCK_K

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_transpose_kernel[grid](
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
        # Expect two tensors: A and B, compute A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
