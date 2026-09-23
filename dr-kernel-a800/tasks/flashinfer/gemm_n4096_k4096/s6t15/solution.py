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
    # 2D tile IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute rows and cols covered by this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A and B tiles
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for boundary checks
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles; cast to fp32
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Manual accumulation over BLOCK_K
        # A_tile: [BLOCK_M, BLOCK_K], B_tile: [BLOCK_K, BLOCK_N]
        for kk in range(0, BLOCK_K):
            # Extract column from A_tile and row from B_tile
            a_col = A_tile[:, kk]        # [BLOCK_M]
            b_row = B_tile[kk, :]        # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]  # [BLOCK_M, BLOCK_N]

    # Store results back to C, cast to original dtype (C has same dtype as input A)
    # We only store where within bounds of M,N
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
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

        # Allocate output tensor (same dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: choose to cover even tiny M,N reliably
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid across M and N tiles
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
