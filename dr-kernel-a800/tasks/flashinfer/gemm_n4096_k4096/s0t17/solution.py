import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small N, moderate M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        # Larger N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        # Mixed
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32},  num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_A_BT_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B[n, k] -> B_T[k, n] via strides
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[offs_m, offs_k]
        A_ptrs = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_T tile: B_T[offs_k, offs_n] = B[offs_n, offs_k]
        BT_ptrs = B + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        BT_tile = tl.load(BT_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Store to C in original dtype (assumed fp16 in provided inputs)
    C_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"

        # Contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"Incompatible shapes: A is [{M}, {K}], B is [{N}, {K2}]"

        # Output tensor with same dtype as A
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B_T[k, n] = B[n, k] -> use B strides directly
        stride_bn = B.stride(0)  # corresponds to n in B
        stride_bk = B.stride(1)  # corresponds to k in B
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch with 2D grid covering tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_A_BT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
