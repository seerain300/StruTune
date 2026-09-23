import triton
import triton.language as tl


@triton.jit
def matmul_bt_kernel(A_ptr, B_ptr, C_ptr,
                     M, N, K,
                     stride_am, stride_ak,
                     stride_bn, stride_bk,
                     stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B.T, where:
      - A: [M, K], row-major
      - B: [N, K], row-major (we index B as B_T[k, n] = B[n, k] using B's strides)
      - C: [M, N], row-major
    Accumulate in fp32, store fp32.
    """
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundary conditions
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Pointers for A[m, k] and B_T[k, n] = B[n, k]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk

        # Load tiles; cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)  # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]

    # Store results
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        """
        Compute C = A @ B.T using Triton.
        - A: [M, K]
        - B: [N, K]
        - Returns C: [M, N] (float32 for stability)
        """
        # Ensure tensors are on CUDA
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton execution."

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Extract shapes
        M, K = A.shape
        N, K2 = B.shape
        # Enforce that B's second dim matches A's K
        assert K == K2, f"B's second dim ({K2}) must match A's second dim ({K})."

        # Allocate output in float32 for numerical stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's original strides:
        stride_bn = B.stride(0)  # first dim of B (N)
        stride_bk = B.stride(1)  # second dim of B (K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose BLOCK sizes based on runtime shapes to ensure full coverage and good perf.
        # These heuristics cover a wide range of sizes without relying on autotune.
        def choose_block(x):
            if x >= 1024:
                return 64
            elif x >= 512:
                return 64
            elif x >= 256:
                return 32
            elif x >= 128:
                return 16
            elif x >= 64:
                return 8
            elif x >= 32:
                return 4
            elif x >= 16:
                return 2
            else:
                return 1

        BLOCK_M = choose_block(M)
        BLOCK_N = choose_block(N)
        # For K, use 32 or 64; 32 balances register usage well for fp32 accumulation
        BLOCK_K = 32 if K >= 32 else 16

        # Grid over M and N tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Return fp32 for robustness; cast to fp16 if strict dtype matching is required.
        return C


def run(*args):
    return ModelNew()(*args)
