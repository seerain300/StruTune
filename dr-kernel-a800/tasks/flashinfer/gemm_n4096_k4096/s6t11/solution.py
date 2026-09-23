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
    # 2D tiling across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices handled by this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Initialize accumulator in fp32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        # k indices for this chunk
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A and B tiles
        # A[m, k] with m = m_offsets[:, None], k = k_offsets[None, :]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B[k, n] with k = k_offsets[:, None], n = n_offsets[None, :]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for out-of-bounds (safe for BLOCK sizes, but keep for generality)
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (dtype follows the tensor's dtype; we'll cast to fp32 for accumulation)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate: sum over BLOCK_K
        # Cast to float32 for stable accumulation
        A_tile_f32 = A_tile.to(tl.float32)
        B_tile_f32 = B_tile.to(tl.float32)

        # Manual reduction over kk
        # For each kk, acc += A_tile[:, kk][:, None] * B_tile[kk, :][None, :]
        for kk in range(0, BLOCK_K):
            # Safe: kk < BLOCK_K; we step over k0 in [0..K] and kk in [0..BLOCK_K-1]
            acc += A_tile_f32[:, kk][:, None] * B_tile_f32[kk, :][None, :]

    # Store back to C, cast to output dtype
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D and contiguous
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose blocks: robust for tiny M/N
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 32

        # Grid: ensure we cover all M and N
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
