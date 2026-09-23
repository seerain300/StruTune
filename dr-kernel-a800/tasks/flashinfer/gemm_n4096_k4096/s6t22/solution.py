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
    # 2D program ids for tiles across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col indices this program will compute
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape [BLOCK_N]

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # shape [BLOCK_K]

        # Masks for bounds
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_tile: we want B[k, n] for k in k_offsets, n in n_offsets
        # B has shape [K, N] with strides (stride_bk, stride_bn)
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Manual accumulation: acc += A_tile @ B_tile
        # A_tile: [BLOCK_M, BLOCK_K], B_tile: [BLOCK_K, BLOCK_N]
        for kk in range(0, BLOCK_K):
            a_col = A_tile[:, kk]  # [BLOCK_M]
            b_row = B_tile[kk, :]  # [BLOCK_N]
            acc += a_col.to(tl.float32)[:, None] * b_row.to(tl.float32)[None, :]

    # Store the result to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=store_mask)


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

        # Output tensor (same dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose blocks that guarantee coverage for tiny M/N (especially M=1)
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 64  # loop handles K=4096 in 64 iterations

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
