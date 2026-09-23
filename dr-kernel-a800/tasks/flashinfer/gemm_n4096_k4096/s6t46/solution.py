import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,      # strides for A (M, K)
    stride_bk, stride_bn,      # strides for B (K, N) — we access B[k, n]
    stride_cm, stride_cn,      # strides for C (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets this program instance will handle
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
        # Pointers for B tile: B[k, n] (we index B with k then n). Note: B is [K, N] so B[k, n] is correct.
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

        # Bounds masks
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)  # [BLOCK_M, BLOCK_K]
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)  # [BLOCK_K, BLOCK_N]

        # Load tiles (fp16 inputs, Triton loads return fp16). We convert to fp32 for accumulation.
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K], fp16
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N], fp16

        A_tile = A_tile.to(tl.float32)  # [BLOCK_M, BLOCK_K]
        B_tile = B_tile.to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += A_tile[:, kk] * B_tile[kk, :] for kk in [0..BLOCK_K-1]
        for kk in range(0, BLOCK_K):
            a_vec = A_tile[:, kk]     # [BLOCK_M]
            b_vec = B_tile[kk, :]     # [BLOCK_N]
            acc += a_vec[:, None] * b_vec[None, :]

    # Store result to C, cast back to fp16 (to match input/output dtype expectation)
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn  # [BLOCK_M, BLOCK_N]
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=store_mask)


class ModelNew(torch.nn.Module):
    def run(self, A, B):
        # Compute C = A @ B.T using Triton. No torch computation except for allocations and .contiguous().
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("A and B must be 2D tensors")
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes for A @ B.T: A is [{M}, {K}], B is [{K2}, {N}]")

        # Ensure contiguous tensors for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Allocate output tensor C: [M, N], same dtype as A (float16 as per original get_inputs)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: small enough to guarantee grid coverage for tiny M/N, and performant for larger sizes
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 32

        # Launch grid over tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Invoke Triton kernel: computes C = A @ B.T
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        # Return C
        return C


def run(*args):
    return ModelNew()(*args)
