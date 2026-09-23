import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M or N
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over output C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        # For each kk within the chunk, do outer-product accumulation
        for kk in range(0, BLOCK_K):
            k = k0 + kk

            # Load A tile: A[m, k], shape [BLOCK_M]
            a_ptrs = A_ptr + (offs_m * stride_am + k * stride_ak)
            a_mask = (offs_m < M) & (k < K)
            a_vec = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load B tile: treat B as [K, N], element (k, n). Using B’s strides: k*stride_bk + n*stride_bn
            b_ptrs = B_ptr + (k * stride_bk + offs_n * stride_bn)
            b_mask = (k < K) & (offs_n < N)
            b_vec = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_N]

            # Outer product accumulation: [BLOCK_M, 1] * [1, BLOCK_N]
            # Triton requires explicit broadcasting; multiply and reduce along the vector dim
            a_vec_f32 = a_vec.to(tl.float32)[:, None]  # [BLOCK_M, 1]
            b_vec_f32 = b_vec.to(tl.float32)[None, :]  # [1, BLOCK_N]
            acc += a_vec_f32 * b_vec_f32

    # Write back to C with mask
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_bT(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using a Triton kernel. A: [M, K], B: [K, N], returns C: [M, N].
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton kernel."
    # Ensure contiguity for coalesced memory access
    A_c = A.contiguous()
    B_c = B.contiguous()

    M, K = A_c.shape
    K2, N = B_c.shape
    assert K == K2, f"Incompatible shapes: A is (*, {K}), B is (*, {N}) with inner dim {K2}, expected K={K}."

    # Allocate output tensor in fp32 for stability
    C = torch.empty((M, N), device=A_c.device, dtype=torch.float32)

    # Compute grid size based on tile sizes
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # Launch kernel
    matmul_bT_kernel[grid](
        A_c, B_c, C,
        M, N, K,
        A_c.stride(0), A_c.stride(1),
        B_c.stride(0), B_c.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original run expects exactly two tensors: A and B
        if len(args) != 2:
            raise RuntimeError("ModelNew.forward expects two input tensors (A, B).")
        A, B = args
        # Ensure both are CUDA tensors; if not, move to current device
        device = torch.device('cuda')
        if not A.is_cuda:
            A = A.to(device)
        if not B.is_cuda:
            B = B.to(device)
        # Run Triton kernel: C = A @ B.T
        C = triton_matmul_bT(A, B)
        return C


def run(*args):
    return ModelNew()(*args)
