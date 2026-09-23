import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,         # A strides for M==1 path (M ignored here but passed for completeness)
    stride_bTr, stride_bTn,       # B_T strides: row stride = B.stride(1), col stride = B.stride(0)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program id along N
    pid = tl.program_id(axis=0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    # Accumulator
    out = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K in chunks
    k0 = 0
    while k0 < K:
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A[0, k_range] (M==1) and cast to fp32
        a_ptrs = A + k_range * stride_ak  # since M == 1, row index is 0
        a = tl.load(a_ptrs, mask=k_range < K, other=0).to(tl.float32)  # [BLOCK_K]

        # Load B_T[k_range, cols] as B[cols, k_range] using B strides (B_T[k, n] = B[n, k])
        b_ptrs = B + cols[None, :] * stride_bTn + k_range[:, None] * stride_bTr  # [BLOCK_K, BLOCK_N]
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0)  # load as fp16, Triton infers type from B
        b = b.to(tl.float32)  # cast to fp32 for accumulation

        # Outer product accumulation: out += sum_k a[k] * b[k, :]
        out += tl.sum(b * a[:, None], axis=0)

        k0 += BLOCK_K

    # Store result to C[0, cols]
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, out.to(C.dtype.element_ty), mask=cols < N)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + k[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.float32)

        # Load B_T tile [BLOCK_K, BLOCK_N] as B[cols, k] with strides (stride_bTn, stride_bTr)
        b_ptrs = B + cn[None, :] * stride_bTn + k[:, None] * stride_bTr
        b_mask = (k[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.float32)

        # Accumulate: acc += a @ b
        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # Store to C
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device; assume GPU inputs as per evaluator
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        M, K = A.shape
        N = B.shape[0]  # since B is [N, K]

        # Output tensor (float16 to match original behavior)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        # For B_T, row stride = B's column stride, col stride = B's row stride
        stride_bTr = B.stride(1)  # along K (columns of B)
        stride_bTn = B.stride(0)  # along N (rows of B)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        if M == 1:
            # Specialized fast path for M == 1: row-wise matvec over N with reduction over K
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bTr, stride_bTn,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic 2D GEMM fallback
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bTr, stride_bTn,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Return C as float16 (matches torch.matmul default output dtype for fp16 inputs)
        return C


def run(*args):
    return ModelNew()(*args)
