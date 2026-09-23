import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_1d_kernel(A_ptr, B_ptr, C_ptr,
                     M: tl.constexpr, N, K,
                     stride_am, stride_ak,
                     stride_bn, stride_bk,
                     stride_cm, stride_cn,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    1D kernel over N tiles. Loops over M inside the kernel with M as constexpr.
    A: [M, K], B: [N, K], C: [M, N]
    Index B as B_T[k, n] = B[n, k], using B's strides.
    """
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for the full M x BLOCK_N
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Build pointers
        # A tile: shape [M, BLOCK_K]
        a_ptrs = A_ptr + (tl.arange(0, M)[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        # B_T tile: shape [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk)

        # Load with masks
        a = tl.load(a_ptrs, mask=(tl.arange(0, M)[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (n_mask[None, :]), other=0.0)

        # Accumulate: (M, BK) @ (BK, BN) -> (M, BN)
        acc += tl.dot(a, b)

    # Store results to C[m, n]
    c_ptrs = C_ptr + (tl.arange(0, M)[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(tl.arange(0, M)[:, None] < M) & (n_mask[None, :]))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2d_kernel(A_ptr, B_ptr, C_ptr,
                     M, N, K,
                     stride_am, stride_ak,
                     stride_bn, stride_bk,
                     stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    General 2D-tiled kernel over M and N. Loops over K.
    A: [M, K], B: [N, K], C: [M, N]
    Index B as B_T[k, n] = B[n, k] via strides.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        # B_T tile: [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk)

        a = tl.load(a_ptrs, mask=(m_mask[:, None]) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (n_mask[None, :]), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(m_mask[:, None]) & (n_mask[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"B's second dim ({K2}) must match A's second dim ({K})."

        # Output tensor in fp32 (accumulate in fp32 is stable)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's original strides:
        stride_bn = B.stride(0)  # first dim of B (N)
        stride_bk = B.stride(1)  # second dim of B (K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # If M is very small, use 1D kernel over N and loop over M inside the kernel.
        # This reduces masked work along M and increases parallelism along N.
        if M <= 8:
            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)
            matmul_1d_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_2d_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )

        # Return fp32 result; if you need fp16, cast here:
        # return C.to(torch.float16)
        return C


def run(*args):
    return ModelNew()(*args)
