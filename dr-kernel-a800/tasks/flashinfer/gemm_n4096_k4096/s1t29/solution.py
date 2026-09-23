import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_BT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute offsets for current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in BLOCK_K chunks (compile-time loop bound via autotune)
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Pointers for A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a = a.to(tl.float32)  # promote to fp32 for accumulation

        # Pointers for B[k, n] tile: shape [BLOCK_K, BLOCK_N] using B's layout (n, k)
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b = b.to(tl.float32)  # promote to fp32 for accumulation

        # Accumulate: acc += A_tile @ B_tile
        acc += tl.dot(a, b)

    # Store results to C[m, n] with proper masking
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, "Incompatible shapes: A is [M, K], B is [N, K]"
        N = N_b

        # Allocate output in fp32 for accumulation, then cast to A.dtype
        C = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)  # stride for N (rows) of B
        stride_bk = B.stride(1)  # stride for K (cols) of B
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid based on tile sizes
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_BT_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn)

        # Cast to match A.dtype
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
