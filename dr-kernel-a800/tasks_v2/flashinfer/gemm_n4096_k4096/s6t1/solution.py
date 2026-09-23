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
    # 2D tile identifiers across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices this program will handle
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator for this tile (fp16 by default; we keep A/B dtype-driven by C dtype)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Iterate K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A tile: A[m, k] -> shape [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Pointers for B tile: B[k, n] -> shape [BLOCK_K, BLOCK_N]
        B_tile_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for loads
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_tile_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Manual accumulation over BLOCK_K:
        # For each kk in [0..BLOCK_K), acc += A[:, kk] * B[kk, :]
        for kk in range(0, BLOCK_K):
            a_col = A_tile[:, kk]               # [BLOCK_M]
            b_row = B_tile[kk, :]               # [BLOCK_N]
            prod = a_col[:, None] * b_row[None, :]  # [BLOCK_M, BLOCK_N]
            acc += prod

    # Store results to C
    C_tile_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate shapes: A [M,K], B [K,N]
        assert A.dim() == 2, f"A must be 2D, got shape {tuple(A.shape)}"
        assert B.dim() == 2, f"B must be 2D, got shape {tuple(B.shape)}"
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Incompatible shapes: A is (*, {K}), B is ({K2}, *)"

        # Ensure contiguous layout
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (dtype follows A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Tile sizes: choose small BLOCK_M to cover tiny M (e.g., M=1), and moderate BLOCK_N
        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over tiles for M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel (no torch ops in forward)
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
