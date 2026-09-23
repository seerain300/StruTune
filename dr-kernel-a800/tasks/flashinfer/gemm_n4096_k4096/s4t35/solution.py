import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A: [M, K]
    stride_bn, stride_bk,   # BT: [N, K] (B.transpose(0, 1))
    stride_cm, stride_cn,   # C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program computes one BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load BT[k, n] tile: shape [BLOCK_K, BLOCK_N], where BT is [N, K]
        bt_ptrs = BT_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        bt_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store C[m, n] as FP16 (matching original inputs)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    We explicitly create BT = B.transpose(0, 1) to use its correct strides in the kernel.
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D tensors"
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Incompatible shapes: A is (*, {K}), B is ({K2}, *).")

    # Ensure CUDA tensors
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)

    # Make A contiguous for simple, coalesced addressing
    A = A.contiguous()

    # BT is a correct non-contiguous view with proper strides
    BT = B.transpose(0, 1)  # BT shape: [N, K]

    # Output tensor (FP16 to match original inputs)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Use conservative tile sizes that previously passed all correctness checks
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, BT, C,
        M, N, K,
        A.stride(0), A.stride(1),
        BT.stride(0), BT.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Original behavior: compute run(A, B) = A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
