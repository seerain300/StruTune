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
    # Each program handles a BLOCK_N chunk of columns for the single row (M == 1)
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    out = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A[0, k_range]
        a_ptrs = A + 0 * stride_am + k_range * stride_ak  # since M == 1, row index is 0
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load B_T[k_range, cols] as B[cols, k_range] using B's strides
        # B_T[k, n] = B[n, k] with row stride = B.stride(1), col stride = B.stride(0)
        b_ptrs = B + cols[None, :] * stride_bTn + k_range[:, None] * stride_bTr  # [BLOCK_K, BLOCK_N]
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate outer product: out += sum_k (a[k] * b[k, :])
        for kk in range(0, BLOCK_K):
            out += a[kk] * b[kk, :]

    # Store result to C[0, :]
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    c_mask = cols < N
    tl.store(c_ptrs, out, mask=c_mask)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTr, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling for general M, N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        km = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + km[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (km[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # B tile interpreted as B_T: [BLOCK_K, BLOCK_N]
        b_ptrs = B + cn[None, :] * stride_bTn + km[:, None] * stride_bTr
        b_mask = (km[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure dtype is float16 as in the original example
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16."
        M, K_a = A.shape
        N_b, K_b = B.shape
        assert K_a == K_b, "A's K must equal B's K."
        K = K_a
        N = N_b

        # Allocate output in float16 (match torch.matmul default behavior here)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Use strides for correct indexing of B as B_T
        stride_am, stride_ak = A.stride(0), A.stride(1)
        # B_T: row stride = B.stride(1), col stride = B.stride(0)
        stride_bTr = B.stride(1)
        stride_bTn = B.stride(0)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        if M == 1:
            # Specialized fast path for M == 1
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
            # Generic GEMM fallback for M > 1
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

        # Return C as is (float16), matching original output dtype for this setup
        return C


def run(*args):
    return ModelNew()(*args)
