import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid is 1D over N. M is assumed == 1 here.
    pid = tl.program_id(axis=0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_cols = cols < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_range < K

        # Load A[0, k_range] as fp32
        a_ptrs = A_ptr + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K], dtype matches A (e.g., fp16)
        a = a.to(tl.float32)  # convert to fp32 for accumulation

        # Load B_T[k_range, cols] = B[cols, k_range] as fp32 tile [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = mask_k[:, None] & mask_cols[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype matches B (e.g., fp16)
        b = b.to(tl.float32)  # convert to fp32 for accumulation

        # Outer product accumulate: acc += sum_k (a[k] * b[k, :])
        # a is [BLOCK_K], b is [BLOCK_K, BLOCK_N]
        acc += tl.sum(b * a[:, None], axis=0)

    # Store to C[m=0, cols] as fp32; C is fp16 tensor so Triton will cast on store
    c_ptrs = C_ptr + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=mask_cols)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A and C
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C and B_T

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        km = k0 + tl.arange(0, BLOCK_K)

        # A submatrix [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rm[:, None] * stride_am + km[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (km[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # B_T submatrix [BLOCK_K, BLOCK_N] = B[cn, km]
        b_ptrs = B_ptr + cn[None, :] * stride_bn + km[:, None] * stride_bk
        b_mask = (km[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store to C[rm, cn]
    c_ptrs = C_ptr + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton; keep dtype as-is
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."

        M, K = A.shape
        N = B.shape[0]  # B is [N, K]; C is [M, N]

        # Output dtype matches A.dtype (torch.matmul preserves dtype)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        if M == 1:
            # Specialized fast path for M == 1
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic fallback for other M
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        return C


def run(*args):
    return ModelNew()(*args)
