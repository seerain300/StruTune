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
    # 2D tile identifiers across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row/col indices this program will handle
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Create pointers to the current tile in A and B
    # A is [M, K], B is [K, N]
    # Pointer for A: A_ptr + m_offsets[:, None]*stride_am + k*stride_ak
    # Pointer for B: B_ptr + k*stride_bk + n_offsets[None, :]*stride_bn

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        # Offsets along K for this chunk
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Build 2D pointers for A_tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (m_offsets[:, None] * stride_am) + (k_offsets[None, :] * stride_ak)
        # Build 2D pointers for B_tile [BLOCK_K, BLOCK_N] (this is B.T loaded)
        B_ptrs = B_ptr + (k_offsets[:, None] * stride_bk) + (n_offsets[None, :] * stride_bn)

        # Masks to avoid out-of-bounds
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles, cast to float32 for robust accumulation
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # dtype: same as A_ptr (fp16), then cast
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)  # dtype: same as B_ptr (fp16), then cast

        # Accumulate: acc += A_tile[:, kk] * B_tile[kk, :]
        # We do this manually to keep robust and avoid tl.dot constraints
        # Note: We can't rely on tl.dot for fp16 here; perform manual FMA over kk
        # Unroll over BLOCK_K (constexpr)
        # Triton will treat BLOCK_K as a compile-time constant
        for kk in range(0, BLOCK_K):
            a_col = A_tile[:, kk]               # [BLOCK_M] in fp16 (cast later)
            b_row = B_tile[kk, :]               # [BLOCK_N] in fp16 (cast later)
            acc += a_col.to(tl.float32)[:, None] * b_row.to(tl.float32)[None, :]

    # Store the result to C with bounds check
    C_ptrs = C_ptr + (m_offsets[:, None] * stride_cm) + (n_offsets[None, :] * stride_cn)
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # C dtype is whatever dtype we allocated (fp16). Triton will cast on store.
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

        # Choose small blocks to guarantee coverage even for tiny M/N
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 32  # iterate in chunks

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
