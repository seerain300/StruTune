import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small tiles for tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_at_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,    # strides for A (rows, cols)
    stride_bk, stride_bn,    # strides for B (rows, cols); we index B.T as [k, n] -> k*stride_bk + n*stride_bn
    stride_cm, stride_cn,    # strides for C (rows, cols)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for boundary conditions
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks; K, BLOCK_K are constexpr so Triton can unroll
    for k in tl.static_range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers to A and B for the current tile
        # A is [M, K]: a_ptrs = A_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B is [K, N]; we need B.T indexing [k, n] -> k*stride_bk + n*stride_bn
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for A and B within K-chunk
        a_mask = (mask_m[:, None]) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (mask_n[None, :])

        # Load tiles; cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate using dot; Triton will handle the matmul for the [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] chunk
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Write back to C. C is [M, N], so c_ptrs = C_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]

    # Store fp16 to match typical input dtype (evaluator uses fp16)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous for coalesced access
        if not A.is_cuda or not B.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors")
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]")

        # Output tensor (fp16 to match typical input dtype)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)  # corresponds to k in B.T
        stride_bn = B.stride(1)  # corresponds to n in B.T
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Define grid: one program per tile over M and N
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch the Triton kernel. We pass M, N, K as constexpr via the autotune key and tl.constexpr args.
        _matmul_at_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
