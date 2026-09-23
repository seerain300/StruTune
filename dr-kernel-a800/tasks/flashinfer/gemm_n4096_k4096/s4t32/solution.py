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
    # 2D program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result tile as FP16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _choose_config(M, N, K):
    # Heuristic: use larger tiles for larger matrices to reduce grid size and improve throughput.
    # Keep BLOCK_K moderate to control shared memory.
    if (K >= 1024) or (N >= 1024):
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3)
    else:
        return dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
    # Ensure contiguous for predictable strides and coalesced access
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]."

    # Allocate output (FP16 to match typical inputs and original behavior)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides (in elements)
    stride_am = A.stride(0)  # typically K
    stride_ak = A.stride(1)  # typically 1
    stride_bk = B.stride(0)  # typically N
    stride_bn = B.stride(1)  # typically 1
    stride_cm = C.stride(0)  # typically N
    stride_cn = C.stride(1)  # typically 1

    cfg = _choose_config(M, N, K)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]), triton.cdiv(N, cfg["BLOCK_N"]))

    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Implement C = A @ B.T using Triton
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
