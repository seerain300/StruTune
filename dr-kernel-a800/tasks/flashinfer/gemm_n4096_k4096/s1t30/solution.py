import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B as if B^T: index B[n, k] using original strides (B is [N, K])
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + (k0 + offs_k[:, None]) * stride_bk)
        b_mask = (offs_n[None, :] < N) & (k_mask[:, None])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Compute C = A @ B.T using Triton
        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, "Inner dimensions must match for matmul"

        # Ensure inputs are contiguous
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output in fp32 for accumulation
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        stride_am, stride_ak = A_c.stride()
        stride_bn, stride_bk = B_c.stride()
        stride_cm, stride_cn = C.stride()

        # Triton grid based on tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_b, meta['BLOCK_N']))

        # Launch the robust 2D-tiled GEMM kernel
        matmul_bT_kernel[grid](
            A_c, B_c, C,
            M, N_b, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast output to match input dtype (get_inputs() uses float16)
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
