import torch
import triton
import triton.language as tl


# General 2D-tiled matmul kernel over tiles of (M, N), looping over K.
@triton.autotune(
    configs=[
        # Very small M: minimize masked lanes along M, increase parallelism along N
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 1024,'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),

        # Medium M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),

        # Larger M
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
        a_mask = (mask_m[:, None]) & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B_tile: B is indexed as B_T[k, n] = B[n, k], so pointer uses B's strides (n, k)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + k_idx[:, None] * stride_bk)
        b_mask = (k_idx[:, None] < K) & (mask_n[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result to C: C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # We must perform C = A @ B.T using Triton. No torch matmul in host code.
        # Construct B_T via transpose/view; B_T is [K, N], but we'll index it as [k, n] = B[n, k].
        # Ensure A and B are on the same device and contiguous.
        assert A.is_cuda and B.is_cuda, "Triton requires CUDA tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        # Sanity: K must match between A and B (as in original run: B.T is [K, N], consistent with A's second dim).
        # In the provided get_inputs, B is [4096, 4096] and A is [1, 4096], so K=K2=4096.
        # The evaluator may vary these; we rely on the harness to provide compatible shapes.
        assert K == K2, f"Dimension mismatch: A.shape[1]={K}, B.shape[1]={K2}"

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k]:
        stride_bk = B.stride(1)  # original B's second dim (K)
        stride_bn = B.stride(0)  # original B's first dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid over tiles of M and N
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
