import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,   # C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: one program computes a tile of size (BLOCK_M, BLOCK_N) in C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C

    # FP32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A[m, k] tile as [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T[k, n] = B[n, k] tile as [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate using fp32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store the accumulated tile to C[m, n]
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # Store as FP16 (matching typical input/output dtype in example). If you need different dtype,
    # you can cast accordingly.
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _select_tiling(M: int, N: int, K: int):
    """
    Simple heuristic for tile selection.
    - For large matrices (K >= 2048 or N >= 2048), use larger tiles and more warps/stages.
    - Otherwise, use moderate tiles.
    """
    if (K >= 2048 or N >= 2048) and (M >= 128):
        return 128, 128, 64, 8, 4
    else:
        return 64, 64, 32, 4, 3


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], C: [M, N].
    All computation is done in Triton kernels. Inputs are made contiguous.
    """
    assert A.is_cuda and B.is_cuda, "Triton kernel requires CUDA tensors"

    # Ensure contiguous tensors for simple stride-based addressing
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]")

    # Output tensor: FP16 to match typical example
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Select tiling and launch config
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _select_tiling(M, N, K)

    # 2D grid over tiles of C
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transpose_tiled_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Implement C = A @ B.T using Triton kernels only.
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors for Triton
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        # Triton computation
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
