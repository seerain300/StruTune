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
    # Tile identifiers across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K] from A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N] from B[k, n] (this accesses B.T pattern)
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Cast to float32 for accumulation
        A_tile = A_tile.to(tl.float32)  # [BLOCK_M, BLOCK_K]
        B_tile = B_tile.to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Manual accumulation over K-chunk
        # acc += sum_{kk in BLOCK_K} A[:, kk] * B[kk, :]
        for kk in range(0, BLOCK_K):
            a_col = A_tile[:, kk]             # [BLOCK_M]
            b_row = B_tile[kk, :]             # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]

    # Store result back to C (Triton will cast to C's dtype if needed)
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"

        # Make tensors contiguous (support non-contiguous via strides, but contiguous is faster)
        A = A.contiguous()
        B = B.contiguous()

        # Extract shapes
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Allocate output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes chosen to guarantee coverage and decent performance
        # Even when M=1 or N=1, grid_m and grid_n will be at least 1.
        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 32

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
            num_warps=4,   # 4 warps per program for 32x128 tile
            num_stages=2,  # pipeline stages for K loop
        )

        return C


def run(*args):
    return ModelNew()(*args)
