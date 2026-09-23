import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_BT_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K using a while loop (runtime K)
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load A tile: A[m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile as if B_T: element at (k, n) of B -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

        k += BLOCK_K

    # Store results to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Shapes
        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, f"Incompatible shapes: A is (*, {K}), B is ({N_b}, {K_b})"

        # Ensure contiguous tensors for performance
        A = A.contiguous()
        B = B.contiguous()

        # Output in fp32 for accumulation, then cast to A.dtype
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Grid based on tiles selected by autotune
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_b, meta['BLOCK_N']))

        matmul_BT_kernel[grid](
            A, B, C,
            M, N_b, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Match original output dtype (inputs are fp16 in get_inputs)
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
