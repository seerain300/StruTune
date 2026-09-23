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
    # 2D tiling identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets handled by this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: A[m, k]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: B[k, n] (we want B.T indexing: k=row, n=col)
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Manual accumulation in float32 for robustness
        A_tile_f32 = A_tile.to(tl.float32)  # [BLOCK_M, BLOCK_K]
        B_tile_f32 = B_tile.to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Unrolled inner loop over BLOCK_K
        for kk in range(BLOCK_K):
            a_col = A_tile_f32[:, kk]       # [BLOCK_M]
            b_row = B_tile_f32[kk, :]       # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]

    # Compute output pointers and store with mask
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)  # acc is float32; Triton will cast to C dtype


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure 2D tensors and contiguous
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor (same dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: ensure coverage even for tiny M/N
        BLOCK_M = 1
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
