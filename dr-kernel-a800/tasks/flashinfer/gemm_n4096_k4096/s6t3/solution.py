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

    # Rows and columns handled by this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Iterate over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A and B tiles
        # A: [M, K], we load rows m_offsets and K columns k_offsets
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B: [K, N], we load rows k_offsets and columns n_offsets (this corresponds to B.T)
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for loads
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)  # [BLOCK_M, BLOCK_K]
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)  # [BLOCK_K, BLOCK_N]

        # Load tiles (dtype inferred from pointers; masks avoid OOB)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Manual accumulation over K-chunk
        for kk in range(BLOCK_K):
            a_col = A_tile[:, kk]  # [BLOCK_M]
            b_row = B_tile[kk, :]  # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]

    # Store results to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)  # [BLOCK_M, BLOCK_N]
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

        # Choose blocks to guarantee coverage even for tiny M/N
        BLOCK_M = 1
        BLOCK_N = 1
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
            num_warps=1, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
