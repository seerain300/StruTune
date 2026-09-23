import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_fused_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A: [M, K]
    stride_bk, stride_bn,   # B: [K, N] (we access B[n, k] logically)
    stride_cm, stride_cn,   # C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B as B[n, k] logically => BT[k, n] tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result as FP16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _choose_launch_params(M, N, K):
    # Heuristic selection of tile sizes and launch parameters
    if (N >= 4096 and K >= 2048):
        return 128, 256, 128, 8, 4
    elif (N >= 2048 or K >= 2048):
        return 128, 128, 64, 8, 3
    elif (N >= 1024 or K >= 1024):
        return 64, 128, 64, 4, 3
    else:
        return 64, 64, 32, 4, 2


def _matmul_transpose_triton(A, B):
    """
    Computes C = A @ B.T using a single Triton kernel. A: [M, K], B: [K, N], returns C: [M, N].
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape  # B is [Kb, N]; typically Kb == K

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)  # along K
    stride_bn = B.stride(1)  # along N
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Choose launch parameters heuristically
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _choose_launch_params(M, N, K)

    # Launch grid: tiles over M and N
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transpose_fused_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two tensors: A and B
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
