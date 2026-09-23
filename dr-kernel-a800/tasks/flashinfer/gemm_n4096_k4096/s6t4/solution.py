import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # strides for A (shape [M, K])
    stride_bk, stride_bn,       # strides for B (shape [K, N]) -- bn is stride over N
    stride_cm, stride_cn,       # strides for C (shape [M, N])
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator (float16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Build pointers for A[m, k] tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Build pointers for B[k, n] tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for loads (handle edge tiles)
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (dtype follows tensor; here A and B are float16)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)   # [BLOCK_M, BLOCK_K], float16
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)   # [BLOCK_K, BLOCK_N], float16

        # Manual accumulation: acc += sum_{kk} A[:, kk] * B[kk, :]
        for kk in range(BLOCK_K):
            a_vec = A_tile[:, kk]     # [BLOCK_M], float16
            b_vec = B_tile[kk, :]     # [BLOCK_N], float16
            acc += a_vec[:, None] * b_vec[None, :]

    # Store results with mask
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate inputs
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("ModelNew.forward expects 2D tensors A and B")

        # Ensure tensors are on CUDA for Triton
        if not A.is_cuda or not B.is_cuda:
            if torch.cuda.is_available():
                A = A.to('cuda')
                B = B.to('cuda')
            else:
                raise RuntimeError("CUDA device required for Triton kernel")

        # Ensure contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Allocate output (float16, same as inputs)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: balanced for performance and robust for small M
        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 64

        # 2D grid over M and N tiles
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

        return C


def run(*args):
    return ModelNew()(*args)
