import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bT_row1_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Specialized kernel for M == 1: compute C[0, :] = A[0, :] @ B.T
    # Grid is 1D along N; we assume grid = (1,) so we process all columns.
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Accumulator vector (fp32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A[0, k] vector: shape [BLOCK_K]
        a_ptrs = A_ptr + (0 * stride_am + offs_k * stride_ak)
        a_vec = tl.load(a_ptrs, mask=mask_k, other=0.0)

        # Load B[k, n] block: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_block = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate dot: [BLOCK_K] dot [BLOCK_K, BLOCK_N] -> [BLOCK_N]
        acc += tl.sum(b_block * a_vec[:, None], axis=0)

    # Store result vector to C[0, :]
    c_ptrs = C_ptr + (0 * stride_cm + offs_n * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_n)


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

        # Larger tiles for big N/K (e.g., N >= 2048, K >= 1024)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bT_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiled kernel for general M,N,K
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # B tile for B.T: [BLOCK_K, BLOCK_N], indexing B[k, n] = k*stride_bk + n*stride_bn
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # A: [M, K], B: [K, N]
        assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is (*, {K}), B is ({Kb}, *), expected K to match"
        # Ensure contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        # Output in fp32 for accumulation, cast to fp16 at the end to match original behavior
        C = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        if M == 1:
            # Specialized fast path: compute entire C[0, :] in one program
            # Use large BLOCK_N to process N in fewer iterations; BLOCK_K moderate to balance registers and ILP.
            grid = (1,)
            matmul_bT_row1_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_N=2048, BLOCK_K=256,
                num_warps=8, num_stages=4,
            )
        else:
            # General tiled kernel; Triton autotune selects best config
            grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
            matmul_bT_tiled_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        # Cast to fp16 to match original run (which uses fp16 inputs)
        if C.dtype != torch.float16:
            C = C.to(torch.float16)
        return C


def run(*args):
    return ModelNew()(*args)
