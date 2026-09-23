import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,  # strides for B_T: (stride along k, stride along n)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a block of columns for row m=0 (M==1 fast path)
    pid = tl.program_id(axis=0)
    col_start = pid * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    out = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k_range] when M == 1
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0)  # shape [BLOCK_K], fp16

        # Load B_T[k_range, cols] as B[cols, k_range] using B strides
        # Note: B_T[k, n] = B[n, k]
        b_ptrs = B + cols[None, :] * stride_bTn + k_range[:, None] * stride_bTr  # [BLOCK_K, BLOCK_N]
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N], fp16

        # Compute outer product for this K-chunk and accumulate
        # Cast to fp32 for stable accumulation
        a32 = a.to(tl.float32)  # [BLOCK_K]
        b32 = b.to(tl.float32)  # [BLOCK_K, BLOCK_N]
        out += tl.sum(b32 * a32[:, None], axis=0)

    # Store results to C row 0
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    # Store as fp16 (C is expected to be fp16)
    tl.store(c_ptrs, out.to(tl.float16), mask=cols < N)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,  # strides for B_T
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C/N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        km = k_start + tl.arange(0, BLOCK_K)  # current K-chunk

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + km[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (km[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile [BLOCK_K, BLOCK_N] as B[cols, k] using B strides
        b_ptrs = B + cn[None, :] * stride_bTn + km[:, None] * stride_bTr
        b_mask = (km[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store acc to C
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Output tensor: match torch.matmul(A, B.T) dtype and device
        M, K = A.shape
        N, Kb = B.shape
        assert K == Kb, "A's second dim must equal B's first dim"
        # Allocate output as float16 to match original inputs' dtype (float16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Extract strides (in elements)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        # For B_T, row stride is B's column stride, col stride is B's row stride
        stride_bTr = B.stride(1)  # along K (columns of B)
        stride_bTn = B.stride(0)  # along N (rows of B)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Fast path for M == 1
        if M == 1:
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

        return C


def run(*args):
    return ModelNew()(*args)
