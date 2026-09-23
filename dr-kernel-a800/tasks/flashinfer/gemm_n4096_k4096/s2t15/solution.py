import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_N': 1024,'BLOCK_K': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 2048,'BLOCK_K': 256}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 4096,'BLOCK_K': 256}, num_warps=16, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_row_kernel(
    A_ptr,      # *fp16, pointer to A of shape [M, K]
    B_ptr,      # *fp16, pointer to B of shape [K, N]
    C_ptr,      # *fp16, pointer to C of shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,        # stride for A on dim 0 (rows)
    stride_ak,        # stride for A on dim 1 (cols)
    stride_bk,        # stride for B on dim 0 (rows)
    stride_bn,        # stride for B on dim 1 (cols)
    stride_cm,        # stride for C on dim 0 (rows)
    stride_cn,        # stride for C on dim 1 (cols)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one output row (m in [0, M))
    m = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)

    # Accumulator for this row (fp32 for numerical stability)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[m, k_offsets] as a vector of length BLOCK_K
        a_row_ptr = A_ptr + m * stride_am
        a_vec = tl.load(
            a_row_ptr + k_offsets * stride_ak,
            mask=k_mask,
            other=0.0
        )  # shape [BLOCK_K], fp16

        # Load B[k_offsets, n_offsets] as a [BLOCK_K, BLOCK_N] tile
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_tile = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] & (n_offsets[None, :] < N)),
            other=0.0
        )  # shape [BLOCK_K, BLOCK_N], fp16

        # Accumulate: dot(a_vec, b_tile, axis=0) -> [BLOCK_N]
        a_vec_f32 = a_vec.to(tl.float32)          # [BLOCK_K]
        b_tile_f32 = b_tile.to(tl.float32)        # [BLOCK_K, BLOCK_N]
        acc += tl.sum(b_tile_f32 * a_vec_f32[:, None], axis=0)

    # Store the accumulated row into C[m, :]
    c_ptrs = C_ptr + m * stride_cm + n_offsets * stride_cn
    tl.store(c_ptrs, acc, mask=(n_offsets < N))


@triton.autotune(
    configs=[
        # Small tiles for tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2d_kernel(
    A_ptr,      # *fp16, pointer to A of shape [M, K]
    B_ptr,      # *fp16, pointer to B of shape [K, N]
    C_ptr,      # *fp16, pointer to C of shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,        # stride for A on dim 0 (rows)
    stride_ak,        # stride for A on dim 1 (cols)
    stride_bk,        # stride for B on dim 0 (rows)
    stride_bn,        # stride for B on dim 1 (cols)
    stride_cm,        # stride for C on dim 0 (rows)
    stride_cn,        # stride for C on dim 1 (cols)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for the tile
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
        )  # [BLOCK_M, BLOCK_K], fp16

        # Load B[k, n] as [BLOCK_K, BLOCK_N] (use B strides: stride_bk, stride_bn)
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_tile = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] & (n_offsets[None, :] < N)),
            other=0.0
        )  # [BLOCK_K, BLOCK_N], fp16

        # Accumulate: acc += a_tile @ b_tile
        a_tile_f32 = a_tile.to(tl.float32)       # [BLOCK_M, BLOCK_K]
        b_tile_f32 = b_tile.to(tl.float32)       # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a_tile_f32, b_tile_f32)    # [BLOCK_M, BLOCK_N]

    # Store the accumulated tile into C
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    mask_m = m_offsets[:, None] < M
    mask_n = n_offsets[None, :] < N
    tl.store(c_ptrs, acc, mask=mask_m & mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguous for coalesced access
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, {N}]"

        # Output tensor (fp16, matching original model behavior)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Heuristic: for very small M, use row-wise kernel; otherwise, use 2D tiled kernel
        # Threshold can be tuned; 16 works well for the provided harness (M=1 typical).
        if M <= 16:
            grid = (M,)
            matmul_row_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )
        else:
            # 2D grid over tiles
            # Choose BLOCK sizes; autotuner will pick best among configs, but grid must be computed here.
            # We can use nominal blocks from configs for grid estimation; Triton handles masking.
            # For generality, use 128x128x64 as nominal, but autotune will override.
            BLOCK_M_nom = 128
            BLOCK_N_nom = 128
            grid = (triton.cdiv(M, BLOCK_M_nom), triton.cdiv(N, BLOCK_N_nom))

            matmul_2d_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
