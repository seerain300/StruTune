import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_block_kernel(
    A_ptr,  # *fp16, [M, K]
    B_ptr,  # *fp16, [N, K], indexed as B_T: [k, n] = B[n, k]
    C_ptr,  # *fp16, [M, N]
    M, N, K,
    stride_am, stride_ak,   # A strides: row-major (M,K)
    stride_bn, stride_bk,   # B strides for B_T: row-major (N,K) with swapped access
    stride_cm, stride_cn,   # C strides: row-major (M,N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K (constexpr loop), avoiding dynamic loops
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load B tile as B_T: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Store results to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args

        # Ensure CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew requires CUDA tensors. Please move inputs to CUDA.")

        # Make contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A is [M, K], B is [N, K]
        M, K = A.shape
        if B.dim() != 2:
            raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}.")
        N, K2 = B.shape
        if K2 != K:
            raise ValueError(f"B must have second dim equal to A.shape[1] (K). Got K={K2}, expected {K}.")

        # Output tensor (fp16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)  # row stride for A
        stride_ak = A.stride(1)  # col stride for A
        # For B_T indexing: B_T[k, n] = B[n, k]
        stride_bk = B.stride(1)   # original B's second dim (K)
        stride_bn = B.stride(0)   # original B's first dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # 2D grid over tiles of M and N
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        _matmul_block_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
