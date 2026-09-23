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
    # 2D tiling across M and N. We set BLOCK_M=1, BLOCK_N=1 to ensure coverage for any M,N.
    pid_m = tl.program_id(0)  # along M
    pid_n = tl.program_id(1)  # along N

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator: for BLOCK_M=1, BLOCK_N=1, this is a 1x1 tile. We keep it general.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A[m, k] and B[k, n]
        # A_ptr + m * stride_am + k * stride_ak -> shape [BM, BK]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B_ptr + k * stride_bk + n * stride_bn -> shape [BK, BN]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for valid loads
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Cast to float32 for accumulation
        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Manual reduction over BLOCK_K:
        # acc += A_tile[:, kk][:, None] * B_tile[kk, :][None, :]
        for kk in range(0, BLOCK_K):
            a_col = A_tile[:, kk]    # [BLOCK_M]
            b_row = B_tile[kk, :]    # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]

    # Store results to C at (m, n). We store with mask to handle boundary.
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    # acc is float32. If A/C are float16, Triton will convert on store.
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor, same dtype as A
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Blocks: choose small to guarantee coverage; 1x1 ensures grid covers all rows/cols
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 32  # chunk size along K

        # Grid across M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # For BLOCK_M=1, this is (M, N)

        # Launch kernel
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
