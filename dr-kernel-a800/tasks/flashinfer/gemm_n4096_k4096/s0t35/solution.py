import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M: small BLOCK_M, moderate BLOCK_N, small BLOCK_K
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 2}, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 2}, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 2}, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 2}, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        # Medium
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 8}, num_stages=2),
        # Larger
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 8}, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # fp32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in chunks
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load B_T tile: B_T[k, n] = B[n, k] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b)  # a: [BM, BK], b: [BK, BN]

    # Store results
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA (Triton requires CUDA). No torch computation here.
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"B's second dim ({K2}) must match A's second dim ({K})."

        # Output tensor in fp32 (accumulate in fp32 for stability)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use B's original strides:
        stride_bk = B.stride(1)  # second dim of B (K)
        stride_bn = B.stride(0)  # first dim of B (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch Triton kernel: 2D grid over M and N tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_2d_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        # If strict fp16 output is required, cast here:
        # return C.to(torch.float16)
        return C


def run(*args):
    return ModelNew()(*args)
