import torch
import triton
import triton.language as tl

# Triton kernel: specialized fast path for M == 1
# Computes C[0, n] = sum_k A[0, k] * B_T[k, n], where B_T[k, n] = B[n, k]
@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid is 1D over N. For M==1 we simply have a single row to process.
    pid_n = tl.program_id(axis=0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for this block of columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    k0 = 0
    while k0 < K:
        k_range = k0 + tl.arange(0, BLOCK_K)
        # Load A[0, k_range] (since M == 1, rm is always 0)
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        # Mask for K tail
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load B_T[k_range, cols] as B[cols, k_range] using B's strides
        b_ptrs = B + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate outer product: acc += sum_k a[k] * b[k, :]
        # Manual reduction over BLOCK_K is fine here
        for kk in range(0, BLOCK_K):
            # Guard: if k_range[kk] >= K, a[kk] is already 0
            acc += a[kk] * b[kk, :]

        k0 += BLOCK_K

    # Store result
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    store_mask = cols < N
    tl.store(c_ptrs, acc, mask=store_mask)


# Triton kernel: generic GEMM for M > 1, computes C[m, n] = sum_k A[m, k] * B_T[k, n]
@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_range = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + k_range[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_range[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # B_T tile: B[rm, k_range] to represent B_T[k_range, rm] (indices swapped)
        b_ptrs = B + rm[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (k_range[:, None] < K) & (rm[None, :] < M)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_M]

        # Fused multiply-add: acc += a @ b (outer products across BLOCK_K)
        # Implement as explicit loop to avoid unsupported operations and keep robustness
        for kk in range(0, BLOCK_K):
            acc += a[:, kk][:, None] * b[kk, :][None, :]

        k0 += BLOCK_K

    # Store C tile
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Compute C = A @ B.T using Triton kernels. All heavy lifting in Triton.
        M, K = A.shape
        N = B.shape[1]  # For B of shape [N, K], N is second dim
        K_a = K

        # Allocate output in fp32 for accumulation
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Fast path for M == 1
        if M == 1:
            BLOCK_N = 256
            BLOCK_K = 128
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K_a,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic path for M > 1
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K_a,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul default behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
