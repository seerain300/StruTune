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
    # 2D tiling identifiers across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows (M) and cols (N) this program handles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Initialize accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        # Address: A[m, k] -> A_ptr + m*stride_am + k*stride_ak
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]

        # Load B_tile: we want B[k, n] (this is B.T indexing). Address: B[k, n] -> B_ptr + k*stride_bk + n*stride_bn
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]

        # Manual accumulation over the K-chunk
        # For each kk in 0..BLOCK_K-1:
        #   acc += a[:, kk][:, None] * b[kk, :][None, :]
        for kk in range(0, BLOCK_K):
            # a[:, kk] is [BM]; b[kk, :] is [BN]
            # Broadcast to [BM, BN]
            acc += a[:, kk][:, None] * b[kk, :][None, :]

    # Store results to C with proper masks
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure 2D inputs and contiguity
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters: ensure grid covers M and N even for tiny sizes
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,  # modest number of warps for decent performance
            num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
