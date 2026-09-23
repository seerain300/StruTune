import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for large K/N
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A strides: [M, K]
    stride_bn, stride_bk,   # B strides: [N, K]
    stride_cm, stride_cn,   # C strides: [M, N]
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

    # Base pointers for the first K tile
    A_ptrs = A + offs_m[:, None] * stride_am + 0 * stride_ak
    B_ptrs = B + offs_n[None, :] * stride_bn + 0 * stride_bk

    # Loop over K dimension in BLOCK_K steps
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Masks for loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)         # [BLOCK_M, BLOCK_K]
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)         # [BLOCK_K, BLOCK_N]

        # Accumulate (cast to fp32 for stability)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

        # Advance to next K tile
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Write back results
    C_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton-only compute; if inputs are not CUDA, fallback to torch to avoid crashes
        if (not A.is_cuda) or (not B.is_cuda):
            return torch.matmul(A, B.T)

        M, K = A.shape
        N, K_b = B.shape
        assert K_b == K, f"Incompatible shapes: A is [{M}, {K}], B is [{N}, {K_b}]"

        # Output in fp32; cast to A.dtype after kernel if needed
        out = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Ensure contiguous for predictable strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Strides (in elements)
        stride_am = A_c.stride(0)
        stride_ak = A_c.stride(1)
        stride_bn = B_c.stride(0)  # B[n, k] => stride along N (rows of B)
        stride_bk = B_c.stride(1)  # stride along K (cols of B)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        # Grid based on autotuned BLOCK_M/N
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        # Launch Triton kernel
        matmul_bt_kernel[grid](
            A_c, B_c, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        # Cast to original input dtype if needed (preserve input dtype)
        if out.dtype != A.dtype:
            out = out.to(A.dtype)
        return out


# Helper functions; ensure inputs are CUDA for Triton
def get_inputs():
    A = torch.randn([1, 4096], dtype=torch.float16, device='cuda')
    B = torch.randn([4096, 4096], dtype=torch.float16, device='cuda')
    return [A, B]


def fused_operator(tensor_0, tensor_1):
    _out = ModelNew()(tensor_0, tensor_1)
    # Return as list for consistency
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
