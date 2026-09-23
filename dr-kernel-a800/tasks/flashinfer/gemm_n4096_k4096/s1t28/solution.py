import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Medium tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Larger N tiles
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_BT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: [row, col]
    stride_bn, stride_bk,   # B strides: [row (n), col (k)]
    stride_cm, stride_cn,   # C strides: [row (m), col (n)]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in BLOCK_K chunks
    # Triton requires static loop bound; K provided as constexpr via autotune configs.
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile as B[n, k] -> B is [N, K], strides (stride_bn, stride_bk)
        # We need B_tile[k, n] for dot: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = n_mask[None, :] & k_mask[:, None]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, f"Incompatible shapes: A is (*, {K}), B is ({N_b}, {K_b})"
        N = N_b

        # Allocate output (fp32 accumulator in kernel, cast later to match A.dtype)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides for A [M, K], B [N, K], C [M, N]
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Launch grid based on tiles (autotune will pick BLOCK sizes, we just set grid here)
        # We use a 2D grid over M and N tiles; Triton will specialize per config.
        # For generality, we compute grid as (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
        # We pass M, N, K so autotune can key the configs appropriately.
        # Note: Triton will substitute BLOCK_M/N from the selected config during compilation.
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))  # initial grid; actual tiles selected by autotune

        matmul_BT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to match A.dtype to keep behavior consistent with original model
        if C.dtype != A.dtype:
            C = C.to(A.dtype)

        return C


def run(*args):
    return ModelNew()(*args)
