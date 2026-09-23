import torch
import triton
import triton.language as tl


# Triton kernel for M == 1: compute Y[0, n] = sum_k A[0, k] * BT[n, k]
# BT is B.T materialized as [N, K] contiguous to avoid stride issues.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr,          # *fp16, shape [1, K]
    BT_ptr,         # *fp16, shape [N, K] contiguous
    Y_ptr,          # *fp16, shape [1, N]
    M, N, K,        # sizes
    stride_am, stride_ak,    # strides for A
    stride_btk, stride_btn,  # strides for BT (contiguous => stride_btn=K, stride_btk=1)
    stride_ym, stride_yn,    # strides for Y
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # fp32 accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[0, k] segment
        A_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a = tl.load(A_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load BT[n, k] tile (shape [BLOCK_N, BLOCK_K])
        BT_ptrs = BT_ptr + n_offsets[:, None] * stride_btn + k_offsets[None, :] * stride_btk
        bt = tl.load(BT_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate dot per column
        acc += tl.sum(bt * a[None, :], axis=1)

    # Store Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    tl.store(Y_ptrs, acc.to(tl.float16), mask=n_mask)


# Generic 2D GEMM: C[M, N] = A[M, K] @ B_T[K, N] where B is [K, N] and B_T[n, k] = B[k, n]
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]; indexing B_T[n, k] uses B[k, n] => stride_bk=N, stride_bn=1
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B_T tile [BLOCK_K, BLOCK_N] using strides
        BT_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(BT_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure dtypes consistent (original uses float16)
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes
        if B.ndim != 2:
            # Defensive fallback
            return torch.matmul(A, B.T)

        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            # Defensive fallback if shapes are incompatible
            return torch.matmul(A, B.T)

        # M == 1 path: use robust Triton kernel with BT materialized and grid tied to BLOCK_N
        if M == 1:
            # Materialize B^T as [N, K] contiguous for simple indexing
            BT = B.t().contiguous()  # shape [N, K], strides (K, 1)
            Y = torch.empty((1, N), dtype=torch.float16, device=A.device)

            def grid(meta):
                # Grid must depend on autotuned BLOCK_N to cover all columns
                return (triton.cdiv(N, meta['BLOCK_N']),)

            _row_matmul_bt_kernel[grid](
                A, BT, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y

        # Generic GEMM for M > 1
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tiling parameters
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _generic_matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(1), B.stride(0),  # B is [K, N]; B_T[n, k] => stride_bn=N, stride_bk=1
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return C


def run(*args):
    return ModelNew()(*args)
