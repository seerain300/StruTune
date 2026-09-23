import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # strides for A: [M, K]
    stride_bk, stride_bn,       # strides for B: [K, N]
    stride_cm, stride_cn,       # strides for C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids across M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: A[m, k] with masks
        A_tile = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )  # shape [BLOCK_M, BLOCK_K], dtype fp16 from ptr

        # Load B tile: B[k, n] with masks (this is B.T[n, k])
        B_tile = tl.load(
            B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0,
        )  # shape [BLOCK_K, BLOCK_N], dtype fp16

        # Accumulate: tl.dot(A_tile [BM,BK], B_tile [BK,BN]) -> [BM,BN] in fp32
        acc += tl.dot(A_tile, B_tile)

    # Store results to C with mask
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D and contiguous
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Allocate output tensor (same device and dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters: chosen to cover small and large sizes robustly
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

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
