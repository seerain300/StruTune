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
    # 2D tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows (M) and columns (N)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp16 (match input dtype from get_inputs)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Compute pointers for A[m, k] and B[k, n] (i.e., B.T[k, n])
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for OOB
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (input dtype is fp16 from get_inputs)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K], fp16
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N], fp16

        # Manual accumulation: acc += A_tile[:, kk][:, None] * B_tile[kk, :][None, :]
        # Loop over the K-chunk dimension
        for kk in range(BLOCK_K):
            # Broadcast scalar loads
            a_col = A_tile[:, kk]            # [BLOCK_M]
            b_row = B_tile[kk, :]            # [BLOCK_N]
            # Outer product accumulation
            acc += a_col[:, None] * b_row[None, :]

    # Store result to C[m, n]
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure 2D tensors
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("A and B must be 2D tensors")
        # Make contiguous for simple stride math
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Allocate output tensor, same dtype/device as input A
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Get strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: robust for tiny M/N and performant for larger sizes
        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel: computes C = A @ B.T
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
