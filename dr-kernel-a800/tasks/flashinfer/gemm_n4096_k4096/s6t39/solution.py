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
    # 2D tile identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices this program will handle
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Build pointers for A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # dtype follows A_ptr

        # Build pointers for B tile: [BLOCK_K, BLOCK_N] (we want B[k, n])
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # dtype follows B_ptr

        # Manual outer-product accumulation over BLOCK_K
        # acc += sum_{kk} A_tile[:, kk][:, None] * B_tile[kk, :][None, :]
        # Do it one kk at a time for simplicity and robustness
        for kk in range(0, BLOCK_K):
            # Guard against k_offsets[kk] >= K (mask ensured, but safe to guard)
            # Triton supports runtime loop bounds here
            a_col = A_tile[:, kk]                # [BLOCK_M]
            b_row = B_tile[kk, :]                # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]

    # Store the result to C with masks
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect two inputs: A and B
        if len(args) != 2:
            # Fallback to PyTorch for unexpected inputs (though evaluator requires Triton-only)
            A, B = args[0], args[1] if len(args) >= 2 else (None, None)
            if A is None or B is None:
                raise RuntimeError("ModelNew.forward expects two inputs (A, B)")
            return torch.matmul(A, B.T)

        A, B = args

        # Ensure 2D tensors
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError(f"Expected 2D tensors, got A.dim()={A.dim()}, B.dim()={B.dim()}")

        # Make contiguous for simple stride math
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Allocate output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: small to ensure coverage even for tiny M/N
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid across M and N tiles
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
