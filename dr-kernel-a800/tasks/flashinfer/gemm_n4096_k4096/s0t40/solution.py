import torch
import triton
import triton.language as tl

# 2D-tiled matmul kernel: A [M, K], B_T [K, N] where B_T[k, n] = B[n, k]
# Accumulate in fp32, store as fp16 (output tensor dtype should be fp16).
@triton.autotune(
    configs=[
        # Very small M: maximize parallelism along N
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 1024, 'BLOCK_K': 32}, num_warps=2, num_stages=3),

        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 512,  'BLOCK_K': 32}, num_warps=4, num_stages=2),

        # Medium M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=2),

        # Larger M
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B tile: [BLOCK_K, BLOCK_N], with B_T[k, n] = B[n, k]
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    # Store results as fp16 (C is fp16)
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Triton will cast to destination dtype as needed
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure 2D and CUDA
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        M, K = A.shape
        N, K2 = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A [{M}, {K}] and B [{N}, {K2}] must have matching second dim.")
        # Construct B_T as a contiguous transposed view for efficient access
        B_T = B.transpose(0, 1).contiguous()  # [K, N]

        # Output as fp16 to match original behavior
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_T.stride(0)  # along K of B_T
        stride_bn = B_T.stride(1)  # along N of B_T
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # 2D grid over tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_kernel[grid](
            A, B_T, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
