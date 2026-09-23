import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small-K oriented configs (balanced, low overhead)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},   num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},   num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4,  num_stages=4),
        # Medium tiles for broader K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=4,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel_smallK(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        # Larger BLOCK_K to reduce loop iterations for big K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel_largeK(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [N, K], output C: [M, N] = A @ B.T
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, "Inner dimension K must match between A and B"

        # Ensure contiguous for predictable strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output in fp32 for accumulation; cast later to A.dtype
        C = torch.empty((M, N_b), device=A_c.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am, stride_ak = A_c.stride()
        stride_bn, stride_bk = B_c.stride()
        stride_cm, stride_cn = C.stride()

        # Choose kernel based on K (heuristic). K <= 256 -> small-K; else large-K.
        def grid_small(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_b, meta['BLOCK_N']))

        def grid_large(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_b, meta['BLOCK_N']))

        if K <= 256:
            matmul_bt_kernel_smallK[grid_small](
                A_c, B_c, C,
                M, N_b, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            matmul_bt_kernel_largeK[grid_large](
                A_c, B_c, C,
                M, N_b, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )

        # Match original dtype
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
