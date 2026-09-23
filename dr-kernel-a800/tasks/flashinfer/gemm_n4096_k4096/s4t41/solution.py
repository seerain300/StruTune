import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,   # C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(0)  # along M dimension
    pid_n = tl.program_id(1)  # along N dimension

    # Offsets for rows (m) and cols (n) of the current tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile corresponding to B[n, k]: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store the result to C (FP16), guarded by bounds
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    # Ensure inputs are contiguous for simple stride arithmetic and coalesced access
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"Inner dimension mismatch: A.shape={A.shape}, B.shape={B.shape}")
    # Output tensor, match dtype (float16 in this setup)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides
    stride_am, stride_ak = A.stride(0), A.stride(1)   # A [M, K]
    stride_bk, stride_bn = B.stride(0), B.stride(1)   # B [K, N]
    stride_cm, stride_cn = C.stride(0), C.stride(1)   # C [M, N]

    # Fixed tiling configuration that previously passed all workloads
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

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
        # Replace original torch.matmul with Triton implementation
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
