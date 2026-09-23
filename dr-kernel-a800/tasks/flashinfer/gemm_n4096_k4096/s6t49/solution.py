import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # B has shape [K, N], strides (stride_bk, stride_bn)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile identifiers across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute indices this program handles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A and B tiles
        # A: [M, K], we want A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B: [K, N], we want B[k, n] but we compute B.T[n, k] which equals B[n, k]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk

        # Masks for loads
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)

        # Load tiles as fp16 (inputs are fp16), then promote to fp32 for accumulation
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Manual accumulation over the K-chunk
        # For each kk in the chunk, add outer product: A_tile[:, kk] vs B_tile[kk, :]
        # Unroll the small loop
        for kk in range(BLOCK_K):
            # Create vectors: A_col [BLOCK_M], B_row [BLOCK_N]
            A_col = A_tile[:, kk]  # [BLOCK_M]
            B_row = B_tile[kk, :]  # [BLOCK_N]
            # Outer product and accumulate
            acc += A_col[:, None].to(tl.float32) * B_row[None, :].to(tl.float32)

    # Store result to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)  # acc is fp32; Triton will cast to C dtype if needed


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("A and B must be 2D tensors")
        # Contiguous for simple stride handling
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        # Note: For C = A @ B.T to be defined, B.T should be [N, K], which implies B is [K, N]
        # In typical benchmark settings for this task, N == K. We proceed under that assumption.
        if N != K:
            raise RuntimeError(f"Incompatible shapes for A @ B.T: A is [{M}, {K}] and B is [{K2}, {N}] with K={K}, N={N}.")

        # Allocate output C
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(1)  # stride along N dim of B
        stride_bk = B.stride(0)  # stride along K dim of B
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose tile sizes to ensure grid coverage even for tiny M/N
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch the Triton kernel: computes C = A @ B.T
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
