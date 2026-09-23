import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large N, large K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
        # Balanced
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        # Small matrices
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_at_btrans_blockptr_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,      # A is [M, K]
    stride_bn, stride_bk,      # B is [N, K]
    stride_cm, stride_cn,      # C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Output pointers
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k0 in range(0, K, BLOCK_K):
        # A tile: [BLOCK_M, BLOCK_K]
        a_block = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),  # iterate K fastest
        )
        a = tl.load(a_block, boundary_check=(0, 1))

        # "B.T" tile: [BLOCK_K, BLOCK_N], accessing B[n, k] directly
        # Note: we want a [BLOCK_K, BLOCK_N] tile with rows = k, cols = n.
        # We create a block pointer over B of shape (N, K) and read at (n, k).
        b_block = tl.make_block_ptr(
            base=B_ptr,
            shape=(N, K),                 # treat B as [N, K]
            strides=(stride_bn, stride_bk),
            offsets=(pid_n * BLOCK_N, k0),  # we want columns n in this tile
            block_shape=(BLOCK_N, BLOCK_K),  # but we need [BLOCK_K, BLOCK_N] view, so we swap
            order=(1, 0),                  # iterate K fastest for loads
        )
        # Load b_block as [BLOCK_K, BLOCK_N]; Triton will interpret the strides accordingly.
        b = tl.load(b_block, boundary_check=(0, 1))

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are contiguous
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K_b = B.shape
        assert K == K_b, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Output in fp32 for numerical stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)  # B is [N, K]
        stride_bk = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid based on selected meta
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        _matmul_at_btrans_blockptr_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Return fp32 output (PyTorch matmul for fp16 also typically returns fp32)
        return C


def run(*args):
    return ModelNew()(*args)
