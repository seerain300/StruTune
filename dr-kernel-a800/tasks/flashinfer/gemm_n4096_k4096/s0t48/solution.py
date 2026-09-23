import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M: small BLOCK_M, large BLOCK_N, moderate BLOCK_K
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 1024, 'BLOCK_K': 32}, num_warps=8, num_stages=3),

        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 128,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=4, num_stages=3),

        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=4, num_stages=3),

        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=4, num_stages=3),

        # Medium M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256,  'BLOCK_K': 64}, num_warps=8, num_stages=3),

        # Larger M
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512,  'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for 2D tiling
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the tile indices
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Masks for boundaries
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load A[m, k]
        a = tl.load(
            A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )

        # Load B_T[k, n] = B[n, k] via strides
        b = tl.load(
            B + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk,
            mask=b_mask,
            other=0.0,
        )

        # Accumulate in fp32
        acc += tl.dot(a, b)

    # Store results to C[m, n]
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(
        C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,  # store fp32; Triton will handle cast if C is fp16
        mask=c_mask,
    )


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs (2D tensors)
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("A and B must be 2D tensors")
        M, K = A.shape
        B_t = B.transpose(0, 1).contiguous()  # [K, N]
        N = B_t.shape[1]

        # Output tensor (fp16 to match original behavior)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_t.stride(0)  # along K
        stride_bn = B_t.stride(1)  # along N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: 2D over tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel
        matmul_kernel[grid](
            A, B_t, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
