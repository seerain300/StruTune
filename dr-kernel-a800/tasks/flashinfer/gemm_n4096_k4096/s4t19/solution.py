import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: shape [M, K] (after .contiguous(): stride_ak=1, stride_am=K)
    stride_bk, stride_bn,   # strides for B: shape [K, N] (after .contiguous(): stride_bn=1, stride_bk=N)
    stride_cm, stride_cn,   # strides for C: shape [M, N] (after .contiguous(): stride_cn=1, stride_cm=N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile row/col indices
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C, index into A
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C, correspond to B^T's cols (n)

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A_tile: A[m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_tile logically as B_T[k, n] = B[n, k] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C (cast to FP16 for output)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _choose_launch_params(M: int, N: int, K: int):
    # Simple heuristic to choose tile sizes and launch config
    # Favor larger tiles for larger problems
    if max(M, N) >= 1024 and K >= 2048:
        return 128, 128, 64, 8, 4
    elif max(M, N) >= 1024 and K >= 1024:
        return 128, 128, 64, 8, 3
    else:
        return 64, 64, 32, 4, 2


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous (simple strides)
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"Incompatible shapes for A and B: A is {A.shape}, B is {B.shape}")

    # Allocate output (FP16 to match original behavior)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _choose_launch_params(M, N, K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # ModelNew must replace the original run(A, B) behavior: compute A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure tensors are on CUDA; the evaluation harness provides CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
