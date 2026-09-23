import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # strides for A: A[m, k]
    stride_bk, stride_bn,     # strides for B: B[k, n]
    stride_cm, stride_cn,     # strides for C: C[m, n]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Pointers for B[k, n] (equivalent to B.T[n, k])
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for boundary conditions
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (masking out-of-bounds with 0)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # Store results to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [K, N]
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: ensure grid covers even tiny M,N
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid over M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
