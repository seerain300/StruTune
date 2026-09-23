import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # General balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile identifiers
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile: [BLOCK_K, BLOCK_N] by indexing B[n, k]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton-only forward: no torch operations
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16, bf16, fp32."
        assert B.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16, bf16, fp32."

        M, K = A.shape
        N_b, K_b = B.shape
        assert K_b == K, f"Incompatible shapes: A is [{M}, {K}], B is [{N_b}, {K_b}]"

        # Output in fp32 for accumulation; cast later to A.dtype
        out = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        # Ensure inputs are contiguous
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Strides
        stride_am = A_c.stride(0)
        stride_ak = A_c.stride(1)
        stride_bn = B_c.stride(0)  # stride along N of B
        stride_bk = B_c.stride(1)  # stride along K of B
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        # Launch grid based on chosen tiles
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N_b, META['BLOCK_N']),
        )

        matmul_bt_kernel[grid](
            A_c, B_c, out,
            M, N_b, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast output to match input A's dtype
        if out.dtype != A.dtype:
            out = out.to(A.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
