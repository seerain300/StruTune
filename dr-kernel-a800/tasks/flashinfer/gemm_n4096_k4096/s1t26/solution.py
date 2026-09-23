import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
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
    # Program IDs for the 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Compute pointers for A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = m_mask[:, None] & k_mask[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a_vals = a_vals.to(tl.float32)

        # Compute pointers for B[n, k] to represent B^T[k, n]
        # We load B[n, k] and use it as if it were B^T[k, n].
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = n_mask[None, :] & k_mask[:, None]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b_vals = b_vals.to(tl.float32)

        # Accumulate
        acc += tl.dot(a_vals, b_vals)

    # Write back to C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on the same device and contiguous
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton."
        # Shapes: A [M, K], B [N, K], output C [M, N]
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's second dim must equal A's second dim, got B: {B.shape}, A: {A.shape}"
        # Make contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Allocate output (fp32 for accumulation, cast to A.dtype at the end)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Launch Triton kernel with dynamic grid based on tile sizes
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        matmul_BT_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )

        # Cast output to match input A dtype (original code uses fp16 inputs by default)
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
