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

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak

        # Pointers for B tile: we want B[k, n], i.e., B[n, k] in its stored layout
        # Since B is [K, N], B[k, n] corresponds to row k, col n. For loading a [BLOCK_K, BLOCK_N] tile,
        # we index B_ptr with k_offsets as rows and n_offsets as cols.
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for loads
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles; promote to float32 for accumulation
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K], dtype from pointer (usually fp16)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], same dtype

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # Do explicit multiply-and-sum to be robust
        # Note: Triton will broadcast correctly; we convert to float32
        for kk in range(0, BLOCK_K):
            A_vec = A_tile[:, kk]           # [BLOCK_M], dtype from A
            B_row = B_tile[kk, :]           # [BLOCK_N], dtype from B
            acc += A_vec.to(tl.float32)[:, None] * B_row.to(tl.float32)[None, :]

    # Store results to C with mask
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [K, N]
        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor
        # Keep C in the same dtype as A; accumulation is in float32 inside the kernel
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)  # for B[k, n], stride along k
        stride_bn = B.stride(1)  # stride along n
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: chosen to cover a wide range of shapes; grid ensures coverage for any M,N
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,  # reasonable for these tile sizes
            num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
