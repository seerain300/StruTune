import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,   'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256,  'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512,  'BLOCK_K': 64},  num_warps=8,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles of C [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator for this tile (fp32 for numerical stability)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load A[m, k] tile as [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_mask[None, :])
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # fp16
        a_tile_f32 = a_tile.to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load B[k, n] tile as [BLOCK_K, BLOCK_N] where B has shape [K, N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_mask[:, None] & (n_offsets[None, :] < N))
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # fp16
        b_tile_f32 = b_tile.to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Manual outer-product accumulation for this K-chunk:
        # For each kk in [0..BLOCK_K), accumulate A[:, k0+kk] * B[k0+kk, :]
        for kk in range(0, BLOCK_K):
            k_idx = k0 + kk
            # a_col: [BLOCK_M] = A[:, k_idx]
            a_col = a_tile_f32[:, kk]
            # b_row: [BLOCK_N] = B[k_idx, :]
            b_row = b_tile_f32[kk, :]
            acc += a_col[:, None] * b_row[None, :]

    # Store the accumulated tile to C (cast to fp16 on store)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def triton_matmul_at_bT(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N], output C: [M, N]
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure contiguous for coalesced access
    A_contig = A.contiguous()
    B_contig = B.contiguous()

    # Output tensor (fp16 to match input dtype in the harness)
    C = torch.empty((M, N), device=A_contig.device, dtype=torch.float16)

    # Triton expects element-wise strides
    stride_am = A_contig.stride(0)
    stride_ak = A_contig.stride(1)
    stride_bk = B_contig.stride(0)
    stride_bn = B_contig.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid: tiles over M and N
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_at_bT_kernel[grid](
        A_contig, B_contig, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Implement C = A @ B.T using Triton; do not use torch.matmul in host code
        if len(args) == 2:
            A, B = args
            return triton_matmul_at_bT(A, B)
        else:
            # Minimal fallback for unusual input counts
            return torch.matmul(args[0], args[1].T)


def run(*args):
    return ModelNew()(*args)
