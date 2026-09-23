import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_smallM_1D_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr,  # specialize for small M; M rows of A are computed at once
    N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # original B strides: first dim (N), second dim (O)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles BLOCK_N columns of the output C
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for all rows m and the BLOCK_N columns
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A_tile for all m rows and current k-block: shape [M, BLOCK_K]
        a_ptrs = A_ptr + (tl.arange(0, M)[:, None] * stride_am) + (k_offsets[None, :] * stride_ak)
        a_tile = tl.load(a_ptrs, mask=(tl.arange(0, M)[:, None] < M) & (k_mask[None, :]), other=0.0).to(tl.float32)  # [M, BLOCK_K]

        # Load B_T for current k-block and BLOCK_N columns: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (n_offsets[None, :] * stride_bn) + (k_offsets[:, None] * stride_bk)
        b_tile = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += a_tile @ b_tile, broadcasting across N columns
        # a_tile shape [M, BLOCK_K], b_tile shape [BLOCK_K, BLOCK_N] -> result [M, BLOCK_N]
        acc += tl.dot(a_tile, b_tile)

    # Store results for all m rows
    for m in range(0, M):
        c_ptrs = C_ptr + m * stride_cm + n_offsets * stride_cn
        tl.store(c_ptrs, acc[m, :].to(tl.float16), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_2D_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # original B strides: first dim (N), second dim (O)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
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

        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Store
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure CUDA tensors and contiguity
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, O = B.shape

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)  # original B's first dim (N)
        stride_bk = B.stride(1)  # original B's second dim (O)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Select kernel based on M
        if M <= 16:
            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)
            matmul_smallM_1D_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_general_2D_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
