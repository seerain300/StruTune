import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_tinyM_kernel(A, B, C,
                         M, N, K,
                         stride_am, stride_ak,
                         stride_bn, stride_bk,
                         stride_cm, stride_cn,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Specialized kernel for very small M. Parallelize primarily along N and vectorize over K.
    Each program computes a BLOCK_N chunk of columns for all M rows by iterating over K in chunks.
    """
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Iterate over M rows; M is tiny, so this loop has few iterations.
    for m in range(0, M):
        # Accumulator for this (m, N-chunk)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Loop over K in chunks
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < K

            # Load A[m, k] as vector of length BLOCK_K
            a = tl.load(A + m * stride_am + k * stride_ak, mask=k_mask, other=0.0)

            # Load B_T[k, n_offsets] = B[n_offsets, k]
            b = tl.load(B + n_offsets * stride_bn + k * stride_bk, mask=n_mask & k_mask, other=0.0)

            # Fused multiply-accumulate along K
            acc += tl.sum(a[:, None] * b[None, :], axis=0)

        # Store results for row m
        tl.store(C + m * stride_cm + n_offsets * stride_cn, acc, mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_kernel(A, B, C,
                          M, N, K,
                          stride_am, stride_ak,
                          stride_bn, stride_bk,
                          stride_cm, stride_cn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    General 2D-tiled GEMM: C[M, N] = A[M, K] @ B_T[K, N], where B_T[k, n] = B[n, k].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K

        # A[m, k]
        a_ptrs = A + m_offsets[:, None] * stride_am + k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # B_T[k, n] = B[n, k]
        b_ptrs = B + n_offsets[None, :] * stride_bn + k[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store
    c_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure inputs are CUDA tensors and 2D
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D."
        M, K = A.shape
        N, O = B.shape

        # For A @ B.T, B's second dim O must equal K (A's second dim)
        if O != K:
            raise RuntimeError(f"Shape mismatch: A is [M, K]={A.shape}, B is [N, O]={B.shape}. For A @ B.T, B's second dim must equal K.")

        # Make inputs contiguous for performance
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor C [M, N], store in fp16 for performance
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's strides:
        stride_bn = B.stride(0)   # original B's first dim (N)
        stride_bk = B.stride(1)   # original B's second dim (O = K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose kernel:
        # - Use tiny-M kernel when M is very small (<= 16).
        # - Use general kernel otherwise.
        if M <= 16:
            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)
            matmul_tinyM_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_general_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
