import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,  # B is [K, N]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices handled by this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # fp16

        # Pointers for B tile: B[k, n] but we need B.T[k, n] = B[n, k]
        # So we index B with (n, k) using its strides.
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bk + k_offsets[:, None] * stride_bn
        B_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # fp16

        # Cast to fp32 for accumulation
        A_tile = A_tile.to(tl.float32)     # [BLOCK_M, BLOCK_K]
        B_tile = B_tile.to(tl.float32)     # [BLOCK_K, BLOCK_N]

        # Manual accumulation over BLOCK_K
        # acc[m, n] += A_tile[m, kk] * B_tile[kk, n]
        for kk in range(BLOCK_K):
            # A[:, kk] shape [BLOCK_M], B[kk, :] shape [BLOCK_N]
            acc += A_tile[:, kk][:, None] * B_tile[kk, :][None, :]

    # Store result to C (C_ptr dtype determines store dtype; acc is fp32)
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("A and B must be 2D tensors")
        # Make inputs contiguous (simple stride handling)
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor (same dtype as input; original uses float16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)   # along K
        stride_bn = B.stride(1)   # along N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes chosen to ensure grid >= 1 even for tiny M/N
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 32

        # Launch grid across M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Invoke Triton kernel: compute C = A @ B.T
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
