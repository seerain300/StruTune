import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for very small M (e.g., M=1)
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_per_output_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output matrix C[M, N]
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    # Compute row and column indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Initialize fp32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # For each output element (m, n), compute dot over this K-chunk
        # We will iterate over BLOCK_K and accumulate into acc.
        # a_vec: [BLOCK_K] for a given m
        # b_vec: [BLOCK_K] by indexing B using k and n: B[k, n] via strides
        for kk in range(BLOCK_K):
            # Mask for the kk-th element
            k_idx = k_offsets[kk]
            k_valid = k_idx < K

            # Load A[m, k_idx] for all m in the tile
            a_ptrs = A_ptr + m_offsets * stride_am + k_idx * stride_ak  # shape [BLOCK_M]
            a_vec = tl.load(a_ptrs, mask=m_mask & k_valid, other=0.0)   # [BLOCK_M]
            a_vec = a_vec.to(tl.float32)

            # Load B[k_idx, n] for all n in the tile (this emulates B.T without transpose)
            b_ptrs = B_ptr + k_idx * stride_bk + n_offsets * stride_bn  # shape [BLOCK_N]
            b_vec = tl.load(b_ptrs, mask=n_mask & k_valid, other=0.0)   # [BLOCK_N]
            b_vec = b_vec.to(tl.float32)

            # Outer-product accumulation: acc += a_vec[:, None] * b_vec[None, :]
            # a_vec: [BLOCK_M], b_vec: [BLOCK_N]
            acc += a_vec[:, None] * b_vec[None, :]

    # Store the result tile to C, casting to fp16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors; original get_inputs uses fp16
        assert A.is_cuda and B.is_cuda, "Triton requires CUDA tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K_b, N = B.shape
        assert K == K_b, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Allocate output tensor with same dtype as inputs (fp16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Compute strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Define grid over M and N tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel
        matmul_bt_per_output_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
