import torch
import triton
import triton.language as tl

# Generic 2D GEMM that computes C = A @ B_T, where B_T is logical [K, N]
# We access B_T[k, n] via B[n, k] using B's strides (no physical transpose needed).
@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]
    stride_bn, stride_bk,   # B is [N, K] but indexed as B_T[k, n] -> B[n, k]
    stride_cm, stride_cn,   # C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        # Pointers for A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile as B_T: [BLOCK_K, BLOCK_N], accessing B[n, k]
        b_ptrs = B + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Specialized fast path for M == 1: 1D grid over N
# Computes C[0, :] where C = A[0, :] @ B_T
@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K], here M==1
    stride_bn, stride_bk,   # B is [N, K], indexed as B_T[k, n] -> B[n, k]
    stride_cm, stride_cn,   # C is [M, N], here M==1
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for this chunk of columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Load A[0, k_range] (M==1, so rm=0)
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0)  # shape [BLOCK_K]

        # Load B_T[k_range, cols] as B[cols, k_range]
        b_ptrs = B + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: out += sum_k a[k] * b[k, :]
        # Implement as dot(a, b): a: [K], b: [K, N] -> [N]
        acc += tl.dot(a, b)

    # Store results
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    c_mask = cols < N
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on the same device
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton"
        # We'll compute in float32 for accuracy and cast back to input dtype at the end.
        M, K = A.shape
        N = B.shape[0]  # B is [N, K]; C is [M, N]
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Fast path for M == 1
        if M == 1:
            # Choose tile sizes tuned for N=4096, K=4096
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
            # Generic path for M > 1
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

        # Cast to original dtype to match torch.matmul default behavior
        return C.to(A.dtype)