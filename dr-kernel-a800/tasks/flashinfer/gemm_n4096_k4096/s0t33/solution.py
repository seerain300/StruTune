import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_2d(
    A_ptr, B_ptr, C_ptr,           # pointers
    M, N, K,                       # sizes: A[M, K], B[N, O], but we use B_T[K, N] with strides
    stride_am, stride_ak,          # A strides: [M, K]
    stride_bk, stride_bn,          # B_T strides: [K, N] where B_T[k, n] = B[n, k]
    stride_cm, stride_cn,          # C strides: [M, N]
    BLOCK_M: tl.constexpr,         # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # fp32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A tile: shape [BLOCK_M, BLOCK_K]
        # A[m, k] at addresses: A_ptr + m*stride_am + k*stride_ak
        a_ptrs = A_ptr + (m_offsets[:, None] * stride_am) + (k_offsets[None, :] * stride_ak)

        # Masks for A
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)

        # Load A tile
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Pointers for B_T tile (B_T[k, n] = B[n, k]): shape [BLOCK_K, BLOCK_N]
        # B_T addresses: B_ptr + k*stride_bk + n*stride_bn
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk) + (n_offsets[None, :] * stride_bn)

        # Masks for B_T
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load B_T tile
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a_tile, b_tile)  # [BLOCK_M, BLOCK_N]

    # Write back to C
    c_ptrs = C_ptr + (m_offsets[:, None] * stride_cm) + (n_offsets[None, :] * stride_cn)
    # Output mask for boundaries
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # Store fp16 (original dtype) for output
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure device/dtype consistency
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"

        M, K = A.shape  # A: [M, K]
        N, O = B.shape  # B: [N, O], but we will use B_T of shape [K, N] where B_T[k, n] = B[n, k]

        # Create B_T as contiguous transpose [K, N]
        B_T = B.transpose(0, 1).contiguous()  # [K, N]
        # Output C: [M, N], dtype fp16
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_T.stride(0)  # original B's second dim (K)
        stride_bn = B_T.stride(1)  # original B's first dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch 2D Triton kernel over tiles
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        matmul_kernel_2d[grid](
            A, B_T, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
