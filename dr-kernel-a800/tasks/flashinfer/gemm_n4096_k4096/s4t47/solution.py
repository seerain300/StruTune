import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_simple_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A: [M, K], contiguous => stride_am=K, stride_ak=1
    stride_bk, stride_bn,   # B: [K, N], contiguous => stride_bk=N, stride_bn=1
    stride_cm, stride_cn,   # C: [M, N], contiguous => stride_cm=N, stride_cn=1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D launch: each program handles a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension; per-k outer-product accumulation
    for k in range(0, K):
        # Load A[m, k] for all m in tile: shape [BLOCK_M]
        a_ptrs = A_ptr + m_offsets * stride_am + k * stride_ak
        a_mask = m_offsets < M
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B[k, n] which equals B_T[n, k] for all n in tile: shape [BLOCK_N]
        b_ptrs = B_ptr + k * stride_bk + n_offsets * stride_bn
        b_mask = n_offsets < N
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate outer product for this k
        acc += a[:, None] * b[None, :]

    # Write back to C as FP16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N], output C: [M, N]
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure CUDA and contiguous
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor (FP16, matching typical inputs)
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides for contiguous tensors
    stride_am, stride_ak = A.stride()  # for [M, K]
    stride_bk, stride_bn = B.stride()  # for [K, N]
    stride_cm, stride_cn = C.stride()  # for [M, N]

    # Grid over tiles
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel with modest warps/stages for robustness
    matmul_transpose_simple_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect A and B as inputs
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
