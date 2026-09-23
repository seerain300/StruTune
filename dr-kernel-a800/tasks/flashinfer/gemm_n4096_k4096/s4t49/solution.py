import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel_per_k(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]: stride_am along rows, stride_ak along cols
    stride_bk, stride_bn,   # B is [K, N]: stride_bk along rows, stride_bn along cols
    stride_cm, stride_cn,   # C is [M, N]: stride_cm along rows, stride_cn along cols
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D program id for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension explicitly
    # Note: For FP16 inputs, we still accumulate in FP32 for better numerical stability.
    # We load per-element and multiply-accumulate, which is robust across any M,N,K.
    for k in range(0, K):
        # Load A[m, k] for all m in this tile
        a_ptrs = A_ptr + m_offsets * stride_am + k * stride_ak
        a_mask = m_offsets < M  # Only M dimension mask needed for A
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_M]

        # Load B[n, k] for all n in this tile (this is B_T[k, n] == B[n, k])
        b_ptrs = B_ptr + n_offsets * stride_bn + k * stride_bk
        b_mask = n_offsets < N  # Only N dimension mask needed for B
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_N]

        # Outer product accumulate: acc[m, n] += A[m, k] * B[n, k]
        # Broadcast a_vals [BLOCK_M] and b_vals [BLOCK_N] to [BLOCK_M, BLOCK_N]
        a_broadcast = a_vals[:, None]
        b_broadcast = b_vals[None, :]
        acc += a_broadcast * b_broadcast

    # Store result to C[m, n] as FP16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    m_mask = m_offsets[:, None] < M
    n_mask = n_offsets[None, :] < N
    store_mask = m_mask & n_mask
    tl.store(c_ptrs, acc.to(tl.float16), mask=store_mask)


def _matmul_transpose_triton(A, B):
    """
    Compute C = A @ B.T using Triton with robust per-k accumulation.
    A: [M, K], B: [K, N], C: [M, N]
    """
    # Ensure CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]"
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Choose conservative tile sizes to minimize masking issues
    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_transpose_kernel_per_k[grid](
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
        # Replace the original run with Triton-based matmul(A, B.T)
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
