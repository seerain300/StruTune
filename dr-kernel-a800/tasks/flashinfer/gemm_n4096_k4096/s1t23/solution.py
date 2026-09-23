import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced tiles for general cases
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Larger N tiles to reduce N-grid size for big N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Larger K tiles to reduce K-loop iterations for large K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        # Very large N tile for massive matrices
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        # Include a few asymmetric tiles to cover edge cases
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D program id over output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], elements A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile as if we had B.T: [BLOCK_K, BLOCK_N], reading B[n, k]
        # Address: B_ptr + n*stride_bn + k*stride_bk
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D CUDA tensors
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution"

        # Make inputs contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, f"Incompatible shapes: A is [{M}, {K}], B is [{N_b}, {K_b}]"
        N = N_b

        # Output in fp32 for stable accumulation; cast to A.dtype after
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Compute strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Grid depends on tile sizes selected by autotune
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_bt_kernel[grid](A, B, C,
                               M, N, K,
                               stride_am, stride_ak,
                               stride_bn, stride_bk,
                               stride_cm, stride_cn)

        # Cast to input A's dtype to match expected behavior
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
