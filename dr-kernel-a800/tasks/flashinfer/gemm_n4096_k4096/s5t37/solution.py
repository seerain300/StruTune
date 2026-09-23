import torch
import triton
import triton.language as tl

# Triton kernel for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * B_T[n, k], where B_T is materialized as [N, K] contiguous.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 512}, num_warps=8, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _matmul_m1_kernel(
    A_ptr,           # *fp16, shape [1, K], row-major
    BT_ptr,          # *fp16, shape [N, K], contiguous (B transposed)
    Y_ptr,           # *fp16, shape [1, N], contiguous
    M, N, K,         # int32
    stride_am, stride_ak,     # strides for A (M, K)
    stride_bnt, stride_btk,   # strides for BT (N, K)
    stride_ym, stride_yn,     # strides for Y (M, N)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # We always have M == 1, so pid_m = 0
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator for the row 0
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A[0, k] vector: shape [BLOCK_K]
        A_row_ptr = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a_vec = tl.load(A_row_ptr, mask=k_offsets < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load BT[n, k] tile: shape [BLOCK_N, BLOCK_K], contiguous in k
        BT_tile_ptr = BT_ptr + n_offsets[:, None] * stride_bnt + k_offsets[None, :] * stride_btk
        mask_tile = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        bt_tile = tl.load(BT_tile_ptr, mask=mask_tile, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate dot product: sum over k dimension -> result [BLOCK_N]
        acc += tl.sum(bt_tile * a_vec[None, :], axis=1)

    # Store result to Y[0, n]
    Y_row_ptr = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    y_mask = n_offsets < N
    tl.store(Y_row_ptr, acc.to(tl.float16), mask=y_mask)


# Triton kernel for general M > 1:
# Computes C[M, N] = A[M, K] @ B_T[K, N] where B_T[n, k] = B[k, n] using B's strides.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_general_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,      # B has shape (K, N); we index as B_T[n, k] = B[k, n]
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

        # A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0).to(tl.float32)

        # B_T tile: [BLOCK_K, BLOCK_N], where B_T[n, k] = B[k, n]
        B_tile_ptr = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a_tile, b_tile)

    # Store result C[m, n] in fp16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    mask_out = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Enforce CUDA tensors for Triton
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Ensure dtypes are float16 for consistency with original code
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). B's K must equal A's K.")

        # Output tensor: Y[M, N], float16, contiguous
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        # M == 1 specialized path: materialize B_T as [N, K] for robust indexing
        if M == 1:
            # BT = B.T contiguous, shape [N, K]
            BT = B.t().contiguous().to(torch.float16)

            # Launch Triton kernel over N tiles; autotune picks BLOCK_N
            grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
            _matmul_m1_kernel[grid](
                A, BT, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
            )
            return Y

        # General path for M > 1: use the GEMM kernel with logical B_T indexing
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        _matmul_general_kernel[grid](
            A, B, Y,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(1), B.stride(0),    # B_T[n, k] = B[k, n] => B strides (N, K)
            Y.stride(0), Y.stride(1),
        )
        return Y


def run(*args):
    return ModelNew()(*args)
