import triton
import triton.language as tl

# Generic 2D GEMM: C[M, N] = A[M, K] @ B_T[K, N] where B_T[k, n] = B[n, k]
@triton.jit
def _matmul_generic_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: (M, K)
    stride_bn, stride_bk,        # B strides: (N, K) but we access B_T[k, n] = B[n, k]
    stride_cm, stride_cn,        # C strides: (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in M
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        # A block: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B_T block: B_T[rk, rn] = B[rn, rk] -> address: B_ptr + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_ptrs = B_ptr + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_mask = (rn[None, :] < N) & (rk[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Fast path for M == 1: compute C[0, :] = sum_k A[0, k] * B_T[k, :]
@triton.jit
def _rowwise_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: (M, K)
    stride_bn, stride_bk,        # B strides: (N, K); we access B_T[k, n] = B[n, k]
    stride_cm, stride_cn,        # C strides: (M, N)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of N columns
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for these columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A[0, k_range] (M==1, so rm=0)
        a_ptrs = A_ptr + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=k_range < K, other=0.0)  # shape [BLOCK_K]

        # Load B_T[k_range, cols] as B[cols, k_range]
        # Address: B_ptr + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_ptrs = B_ptr + cols[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (k_range[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Outer product accumulate: acc += sum_k a[k] * b[k, :]
        acc += tl.sum(b * a[:, None], axis=0)

    # Store to C[0, cols]
    c_ptrs = C_ptr + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=cols < N)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure dtype and device consistency; we compute in fp32, then cast back
        M, K_a = A.shape
        N, K_b = B.shape
        assert K_a == K_b, "A's K must match B's K"
        K = K_a

        # Output in fp32 for accumulation accuracy; cast later to A.dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Optimize for M == 1: parallelize across N, loop K in chunks
            BLOCK_N = 128  # tile size along N
            BLOCK_K = 256  # tile size along K
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),  # B_T[k, n] = B[n, k]
                C.stride(0), C.stride(1),
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
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),  # B_T[k, n] = B[n, k]
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Match original model's output dtype (inputs are fp16)
        return C.to(A.dtype)