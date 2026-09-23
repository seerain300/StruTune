import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid: each program handles a chunk of columns
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    # Accumulator for the BLOCK_N columns handled by this program
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_range < K

        # Load A[0, k_range] (M == 1 specialization)
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K], fp16

        # Load BT[k_range, cols] as a matrix tile [BLOCK_K, BLOCK_N]
        b_ptrs = BT + k_range[:, None] * stride_bTk + cols[None, :] * stride_bTn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N], fp16

        # Cast to fp32 for accumulation
        a = a.to(tl.float32)          # [BLOCK_K]
        b = b.to(tl.float32)          # [BLOCK_K, BLOCK_N]

        # Outer product accumulate: acc += sum_k (a[k] * b[k, :])
        # Equivalent to acc += tl.sum(b * a[:, None], axis=0)
        acc += tl.sum(b * a[:, None], axis=0)

    # Store results to C[0, cols]
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A/C
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C/BT

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        km = k0 + tl.arange(0, BLOCK_K)

        # Load A[rm, km] -> [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + km[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (km[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load BT[km, cn] -> [BLOCK_K, BLOCK_N]
        b_ptrs = BT + km[:, None] * stride_bTk + cn[None, :] * stride_bTn
        b_mask = (km[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C[rm, cn]
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [N, K]
        M, K = A.shape
        N = B.shape[0]  # B is [N, K]

        # Explicitly create B_T with correct shape and strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output in fp32 for stability; cast back at end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Fast path for M == 1
        if M == 1:
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic 2D GEMM for other M
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
