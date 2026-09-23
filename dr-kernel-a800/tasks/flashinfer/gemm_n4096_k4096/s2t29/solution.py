import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_b_t_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,      # B is [K, N]; B_T[k, n] = B[n, k] -> offset = n*stride_bn + k*stride_bk
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id over tiles in M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the current output tile
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k  # current K indices for this chunk

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile emulating B.T: [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + n[None, :] * stride_bn + k[:, None] * stride_bk
        b_mask = (k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b)

    # Store results
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguity for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Inner dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"

        # Output tensor: fp16 to match typical harness expectations
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)  # stride for K (rows of B)
        stride_bn = B.stride(1)  # stride for N (cols of B)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid: 2D over tiles in M and N
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_b_t_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
