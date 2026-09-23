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
    # 2D program ids for tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices this program handles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_mask[None, :])
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype inferred from A

        # Load B[k, n] tile: shape [BLOCK_K, BLOCK_N]
        # Note: we index B by k then n, which corresponds to B.T
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_mask[:, None]) & (n_offsets[None, :] < N)
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype inferred from B

        # Accumulate: acc += A_tile @ B_tile
        # Manual FMA over BLOCK_K
        for kk in range(BLOCK_K):
            a_col = A_tile[:, kk]        # [BLOCK_M]
            b_row = B_tile[kk, :]        # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]

    # Store the result to C[m, n]
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # Cast to output dtype (C is allocated with desired dtype)
    # Triton will implicitly cast on store if C_ptr dtype differs; here we keep C as float32 for stability.
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate shapes
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("A and B must be 2D tensors")
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor: same dtype as A, but kernel accumulates in float32
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters: chosen to ensure coverage and decent performance
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over tiles in M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,  # 4 warps per program is a good default for these tile sizes
            num_stages=2,
        )

        # If original expected dtype is float16, cast back
        if C.dtype != A.dtype:
            C = C.to(A.dtype)

        return C


def run(*args):
    return ModelNew()(*args)
