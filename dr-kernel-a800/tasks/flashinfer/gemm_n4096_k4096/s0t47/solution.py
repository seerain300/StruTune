import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M, large N: prioritize parallelism along N
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 1024,'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 1024,'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 1024,'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 1024,'BLOCK_K': 16}, num_warps=4, num_stages=2),
        # Medium M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        # Larger M
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B_t, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile ids along M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=a_mask, other=0.0)

        # Load B_t tile [BLOCK_K, BLOCK_N], where B_t[k, n] = B[n, k]
        b = tl.load(B_t + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C (cast to fp16)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure 2D inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K = A.shape

        # Create B_T as [K, N] via layout change (not computation)
        B_t = B.transpose(0, 1).contiguous()  # shape [K, N]
        N = B_t.shape[1]

        # Output tensor (fp16 to match typical behavior)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_t.stride(0)  # along K
        stride_bn = B_t.stride(1)  # along N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid over tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_kernel[grid](
            A, B_t, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
