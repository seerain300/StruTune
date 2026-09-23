import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_transposed_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,     # strides for A (shape [M, K])
    stride_bk, stride_bn,     # strides for B (shape [K, N])
    stride_cm, stride_cn,     # strides for C (shape [M, N])
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for the tile (fp32 for numerical stability)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[m, k] as [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_tile = tl.load(
            a_ptrs,
            mask=(m_offsets[:, None] < M) & (k_mask[None, :]),
            other=0.0
        )  # [BLOCK_M, BLOCK_K], dtype of A (fp16/half)

        # Load B[k, n] as [BLOCK_K, BLOCK_N] (B has shape [K, N])
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_tile = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] & (n_offsets[None, :] < N)),
            other=0.0
        )  # [BLOCK_K, BLOCK_N], dtype of B

        # Accumulate: acc += a_tile @ b_tile using fp32
        a_tile_f32 = a_tile.to(tl.float32)   # [BLOCK_M, BLOCK_K]
        b_tile_f32 = b_tile.to(tl.float32)   # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a_tile_f32, b_tile_f32)  # [BLOCK_M, BLOCK_N]

    # Store the accumulated tile into C (C is allocated as fp16 in host)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure inputs are 2D and contiguous for coalesced access
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]"

        # Allocate output C as fp16 to match typical baseline behavior for fp16 inputs
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Compute strides in elements
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = C.stride()

        # Grid over tiles; Triton autotuner will select best config
        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N, meta['BLOCK_N']),
            )

        matmul_at_transposed_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
