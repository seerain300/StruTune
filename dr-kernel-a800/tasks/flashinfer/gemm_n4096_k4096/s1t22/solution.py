import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller tiles for small matrices
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        # Medium tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Larger N tiles to reduce N-grid size when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8,  num_stages=4),
        # Very large N tiles for very large N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128},  num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128},  num_warps=16, num_stages=4),
        # Include some K=256 variants for very large K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 256},  num_warps=8,  num_stages=4),
    ],
    key=['M', 'N_b', 'K_a', 'K_b'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N_b, K_a, K_b,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over C[M, N_b]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K_a, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k] -> [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_a)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile as B[n, k] -> [BLOCK_K, BLOCK_N] by addressing B[n, k] with original strides
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N_b) & (offs_k[:, None] < K_b)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: acc += A_tile @ B_T_block
        acc += tl.dot(a, b)

    # Store result C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_b)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        M, K_a = A.shape
        N_b, K_b = B.shape

        # Output in fp32 for stability; cast after kernel to match input dtype if needed
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)
        stride_bk = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid based on selected BLOCK sizes
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_b, meta['BLOCK_N']))

        matmul_bt_kernel[grid](
            A, B, C,
            M, N_b, K_a, K_b,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to A.dtype to match expected output dtype
        if C.dtype != A.dtype:
            C = C.to(A.dtype)

        return C


def run(*args):
    return ModelNew()(*args)
