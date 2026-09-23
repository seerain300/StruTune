import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: A is [M, K]
    stride_bk, stride_bn,   # strides for B: B is [K, N]
    stride_cm, stride_cn,   # strides for C: C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program handles a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile indices
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C (index into A)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C (correspond to B^T's columns)

    # FP32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T[k, n] = B[n, k] tile: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result in FP16 (to match original dtype behavior)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure 2D inputs
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2D tensors")
    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"Incompatible shapes: A is [M, K]={A.shape}, B is [Kb, N]={B.shape}")

    # Make contiguous and cast to float16 for Triton (original example uses float16)
    A = A.contiguous().to(torch.float16)
    B = B.contiguous().to(torch.float16)

    # Allocate output tensor
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Choose fixed tiling for robustness (previously verified)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

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
        # Replace the original run with Triton-based matmul(A, B.T)
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
