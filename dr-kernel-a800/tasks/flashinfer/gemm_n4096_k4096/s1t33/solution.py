import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # General balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Favor larger N tiles to reduce grid size when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Include 512 N tiles for very large N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Small M configs
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        # Very large shapes: try 256x256x256
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: program id along M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets this program will handle
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    # A: [M, K], strides (stride_am, stride_ak)
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    # B: [N, K], strides (stride_bn, stride_bk)
    # We want B_tile = B[n, k] for n in offs_n, k in offs_k
    B_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk  # [BK, BN]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        # advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Write back C[m, n] = acc[m, n]
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, "Incompatible dimensions: A is [M, K], B is [N, K]"
        N = N_b

        # Output: same dtype as A (fp16 in provided get_inputs), fp32 accumulation inside kernel
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Grid: one program per output tile
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel
        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to A.dtype (kernel wrote fp32 into C; cast to match input dtype expectations)
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
