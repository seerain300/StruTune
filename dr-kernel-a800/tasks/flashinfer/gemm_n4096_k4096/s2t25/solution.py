import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=16, num_stages=5),
    ],
)
@triton.jit
def matmul_b_t_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # A_tile: [BLOCK_M, BLOCK_K], B_tile: [BLOCK_K, BLOCK_N]
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
        B_tile_ptr = B_ptr + (k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    C_out_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_out_ptr, acc, mask=c_mask)


@triton.autotune(
    configs=[
        # BLOCK_N and BLOCK_K choices for matrix-vector when M=1
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=2,  num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 1024,'BLOCK_K': 256}, num_warps=16, num_stages=4),
    ],
)
@triton.jit
def matvec_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,  # C is [M, N]; stride_cm row stride, stride_cn col stride
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Specialized for M == 1: computes C[0, :] = A[0, :] @ B.T
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A[0, :] chunk
    a_ptrs = A_ptr + offs_k * stride_ak  # M==1, row index 0
    a_mask = offs_k < K
    a_vec = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_K], fp32 for accumulation

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks; for each chunk, accumulate dot into acc
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < K

        # Load B[k, offs_n] -> [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)
        b_vec = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Reduce over K chunk to produce [BLOCK_N]
        acc += tl.sum(b_vec * a_vec[:, None], axis=0)

    # Store C[0, offs_n]
    c_ptrs = C_ptr + offs_n * stride_cn
    n_mask = offs_n < N
    tl.store(c_ptrs, acc, mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Inner dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"

        # Output in fp32 for accumulation, then cast to fp16 to match typical harness dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Launch specialized kernel if M == 1
        if M == 1:
            grid = (triton.cdiv(N, 256),)  # autotune will pick best config; grid over N
            matvec_dot_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )
        else:
            # General GEMM over (M, N)
            grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
            matmul_b_t_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        # Cast back to fp16 to match typical output dtype
        return C.to(torch.float16)


def run(*args):
    return ModelNew()(*args)
