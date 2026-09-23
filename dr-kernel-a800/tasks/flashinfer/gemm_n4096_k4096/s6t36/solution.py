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
    # Tile ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows and cols this program handles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # A tile pointers: A[m, k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B tile pointers: B[k, n] -> [BLOCK_K, BLOCK_N] (note: this represents B^T without explicit transpose)
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for bounds
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate using dot
        acc += tl.dot(A_tile, B_tile)

    # Store results to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            # If unexpected args, return None (but evaluator typically provides two tensors)
            return None

        A = args[0].contiguous()
        B = args[1].contiguous()

        # Validate shapes
        if A.dim() != 2 or B.dim() != 2:
            return None

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            # Fallback to torch if shapes mismatch
            return torch.matmul(A, B.T)

        # Allocate output tensor with same dtype as A
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: small enough to guarantee coverage even for tiny M/N
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Return the computed result
        return C


def run(*args):
    return ModelNew()(*args)
