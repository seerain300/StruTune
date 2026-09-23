import torch
import triton
import triton.language as tl

@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a block of columns for a single row m=0 (M==1 fast path).
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for [BLOCK_N] columns
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K dimension in chunks
    for k in range(0, K, BLOCK_K):
        k_range = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A[0, k_range] -> shape [BLOCK_K]
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load B_T[k_range, cols] interpreted as B[cols, k_range]
        # B_T stride: row stride = B.stride(1), col stride = B.stride(0)
        b_ptrs = B + cols[None, :] * stride_bTn + k_range[:, None] * stride_bTr  # [BLOCK_K, BLOCK_N]
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Outer product accumulate: acc += sum_k a[k] * b[k, :]
        # a is [BLOCK_K], b is [BLOCK_K, BLOCK_N]
        for kk in range(0, BLOCK_K):
            acc += a[kk] * b[kk, :]

    # Store results to C[0, cols] in float32 (C is float32 in host-side allocation)
    out_ptrs = C + 0 * stride_cm + cols * stride_cn
    out_mask = cols < N
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N, reducing over K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A/C
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C/N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + kk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (kk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # B_T tile: B_T[kk, cn] = B[cn, kk] -> [BLOCK_K, BLOCK_N]
        b_ptrs = B + cn[None, :] * stride_bTn + kk[:, None] * stride_bTr
        b_mask = (kk[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # acc += A_tile @ B_tile
        # Use broadcasting: a [BLOCK_M, BLOCK_K], b.T [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # Store C[rm, cn]
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N_B, K_B = B.shape
        assert K == K_B, "Inner dimension K must match for A and B"
        N = N_B

        # Output as float32 for accumulation stability, then cast to A.dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        stride_am, stride_ak = A.stride(0), A.stride(1)
        # For B_T, row stride is B's column stride, col stride is B's row stride
        stride_bTr = B.stride(1)  # along K (columns of B)
        stride_bTn = B.stride(0)  # along N (rows of B)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        if M == 1:
            # Specialized fast path for M == 1: focus parallelism along N
            BLOCK_N = 256
            BLOCK_K = 512  # reduce K-loop iterations to 8 for K=4096
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
            # Generic 2D GEMM fallback for M > 1
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

        # Cast to original dtype to match torch.matmul default behavior
        return C.to(A.dtype)