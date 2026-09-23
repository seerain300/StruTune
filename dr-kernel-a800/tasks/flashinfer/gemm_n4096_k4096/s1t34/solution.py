import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # General balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Favor larger N tiles when N is large
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Favor larger N tiles (512) for very large N
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64},   num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        # Small-M oriented configs
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        # Deeper pipelining for long K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256},  num_warps=8, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tile: B is [N, K]; we want B^T so we access B[n, k] with n along N and k along K.
        # Pointer layout: (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store result to C
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure 2D inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        # Ensure same dtype and supported
        assert A.dtype == B.dtype, "A and B must have the same dtype"
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype"

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N = B.shape[1]  # B is [N, K]

        # Output in fp32 for accumulation; cast to input dtype at the end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides in elements
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Grid based on tile sizes
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to original input dtype to match baseline behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
