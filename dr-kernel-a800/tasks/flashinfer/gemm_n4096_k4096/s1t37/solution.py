import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for general GEMM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=16, num_stages=4),
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
    # Tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices for C
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # col indices for C

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K tiles
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile accessed as B.T: shape [BLOCK_K, BLOCK_N], addressing B[offs_n[j], offs_k[i]]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate (inputs may be fp16/bf16/fp32; cast to fp32 for accumulation)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Write back C: [M, N]
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA for Triton execution
        if not (A.is_cuda and B.is_cuda):
            A = A.to('cuda', non_blocking=True)
            B = B.to('cuda', non_blocking=True)

        # Shapes
        M, K = A.shape
        N_b, K_b = B.shape
        if K_b != K:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}], B is [{N_b}, {K_b}]")

        # Output tensor in fp32 (accumulation), will cast to match input dtype after
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        # Ensure inputs are contiguous
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Strides (in elements)
        stride_am = A_c.stride(0)
        stride_ak = A_c.stride(1)
        stride_bn = B_c.stride(0)  # stride along B's "N" (row) dimension
        stride_bk = B_c.stride(1)  # stride along B's "K" (col) dimension
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Dynamic grid based on meta tile sizes
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N_b, META['BLOCK_N']),
        )

        # Launch Triton kernel; no torch matmul in forward
        matmul_bt_kernel[grid](
            A_c, B_c, C,
            M, N_b, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast output to match A.dtype (usually float16)
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
