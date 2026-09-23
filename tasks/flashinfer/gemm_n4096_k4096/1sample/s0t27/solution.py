import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B has strides (bn = stride along N, bk = stride along K)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a block of N columns; M is expected to be 1 for this kernel
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    # Accumulator for this row-block
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_range < K

        # Load A[0, k_range] as a vector (M == 1 -> rm = 0)
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        # Load B_T[k_range, cols] as B[cols, k_range] -> address: B + cols*stride_bn + k_range*stride_bk
        b_ptrs = B + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = mask_n[None, :] & mask_k[:, None]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate outer products: acc += sum_k a[k] * b[k, :]
        # Equivalent to acc += tl.sum(b * a[:, None], axis=0)
        acc += tl.sum(b * a[:, None], axis=0)

    # Store results to C[0, cols]
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # rows of C
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # cols of C
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        # A[m, k] load: A + rm[:, None]*stride_am + rk[None, :]*stride_ak
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)  # [BLOCK_M, BLOCK_K]

        # B_T[k, n] load as B[n, k]: B + rn[None, :]*stride_bn + rk[:, None]*stride_bk
        b_ptrs = B + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=(rn[None, :] < N) & (rk[:, None] < K), other=0.0)  # [BLOCK_K, BLOCK_N]

        # acc += a @ b
        acc += tl.dot(a, b)

    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate dtypes
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise AssertionError("A dtype must be a floating type.")
        if B.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise AssertionError("B dtype must be a floating type.")

        # Shapes: A is [M, K], B is [N, K] -> C = A @ B.T is [M, N]
        M, K = A.shape
        N = B.shape[0]

        # Output in fp32 for numerical stability; cast to A.dtype afterwards
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose fast path for M == 1
        if M == 1:
            # Tile sizes tuned for N=4096, K=4096
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

        # Cast back to original dtype to match torch.matmul behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
