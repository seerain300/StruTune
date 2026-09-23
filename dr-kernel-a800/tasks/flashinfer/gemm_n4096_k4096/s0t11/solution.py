import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Very small M: prioritize parallelism along N, small tiles along M
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        # Small/medium M
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        # Larger M, potentially large N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=16, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_aten_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B_T tile: [BLOCK_K, BLOCK_N], B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result C[m, n] as fp16
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args

        # Device check
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew requires CUDA tensors. Please move inputs to CUDA.")

        # Ensure contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A is [M, K], B is [N, O]; we index B as B_T without materializing
        M, K = A.shape
        if B.dim() != 2:
            raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}.")
        N, _O = B.shape

        # Output tensor (fp16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k]: k-index is B.stride(1), n-index is B.stride(0)
        stride_bk = B.stride(1)
        stride_bn = B.stride(0)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid over M and N tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_aten_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
