import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp16 (inputs are fp16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak  # [BM, BK]
        # Pointers for B[k, n] (we use as-is, indexing as B[k, n])
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn  # [BK, BN]

        # Masks for loads
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)  # [BM, BK]
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)  # [BK, BN]

        # Load tiles as fp16
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BM, BK], fp16
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BK, BN], fp16

        # Manual reduction over BLOCK_K: acc += A_tile[:, kk] * B_tile[kk, :]
        for kk in range(0, BLOCK_K):
            a_col = A_tile[:, kk]   # [BM], fp16
            b_row = B_tile[kk, :]   # [BN], fp16
            acc += a_col[:, None] * b_row[None, :]

    # Store result to C with masks
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        assert A.shape[1] == B.shape[0], "Inner dimension must match: A.shape[1] == B.shape[0]"

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Incompatible shapes"

        # Allocate output tensor (same dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling. Small tiles to guarantee coverage even for tiny M/N.
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 32  # chunk size over K

        # Grid across M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
