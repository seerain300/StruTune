import triton
import triton.language as tl


# Specialized fast kernel for M == 1: computes C[0, n] = sum_k A[0, k] * B.T[k, n]
# We pass B's strides and explicitly interpret them as B.T's strides:
#   stride_bT_rows = B.stride(1)  (original B's stride along K)
#   stride_bT_cols = B.stride(0) (original B's stride along N)
@triton.jit
def _matmul_rowwise_A1_BTK_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,          # A strides: A is [M, K]
    stride_bT_rows, stride_bT_cols,  # B.T strides: B.T is [K, N], where B is [N, K]
    stride_cm, stride_cn,            # C strides: C is [M, N]
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program id along N (columns)
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # initialize accumulator (M=1 means one row in C)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # load A row segment (M=1, so pointer += 0 * stride_am)
        a = tl.load(A_ptr + 0 * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0).to(tl.float32)  # [BLOCK_K]
        # load B.T tile: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bT_rows + offs_n[None, :] * stride_bT_cols
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]
        # outer product accumulation: acc += sum_k a[k] * b[k, :]
        acc += tl.sum(b * a[:, None], axis=0)

    # store result to C[0, offs_n]
    tl.store(C_ptr + 0 * stride_cm + offs_n * stride_cn, acc, mask=offs_n < N)


# Generic GEMM kernel for A[M, K] @ B_T[K, N] -> C[M, N]
@triton.jit
def _matmul_generic_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,          # A strides
    stride_bT_rows, stride_bT_cols,  # B.T strides (from original B)
    stride_cm, stride_cn,            # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_init
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        # B.T tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bT_rows) + (offs_n[None, :] * stride_bT_cols)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        # Fused multiply-add
        acc += tl.dot(a, b)

    # Write back C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on CUDA and contiguous
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        N = B.shape[1]  # for B.T, number of columns equals original B's number of columns

        # Output tensor: [M, N], float32 accumulation and output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides for A, C
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # For B.T, derive strides from original B (shape [N, K]):
        # B.T is [K, N], so:
        stride_bT_rows = B.stride(1)  # along original B's K dimension
        stride_bT_cols = B.stride(0)  # along original B's N dimension

        if M == 1:
            # Optimized row-wise kernel: 1D grid over N
            BLOCK_N = 256
            BLOCK_K = 128
            grid = (triton.cdiv(N, BLOCK_N),)
            _matmul_rowwise_A1_BTK_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bT_rows, stride_bT_cols,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic 2D GEMM kernel
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bT_rows, stride_bT_cols,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
        return C