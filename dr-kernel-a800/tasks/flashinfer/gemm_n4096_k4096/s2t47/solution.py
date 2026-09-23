import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles for tiny M or moderate N
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),

        # Larger column tiles for big N
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_rowwise_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes one row m and a tile of columns [n_start, n_start + BLOCK_N)
    m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Accumulator for this row-tile (fp32)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A[m, offs_k] as a vector (fp16), accumulate in fp32
        a_ptrs = A_ptr + m * stride_am + offs_k * stride_ak
        a_mask = offs_k < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B[offs_k, offs_n] as a [BLOCK_K, BLOCK_N] tile (fp16), accumulate in fp32
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k a[k] * b[k, :]
        # Broadcast a over N, b over K, reduce along K axis
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store results to C[m, offs_n]
    c_ptrs = C_ptr + m * stride_cm + offs_n * stride_cn
    c_mask = offs_n < N
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_rowwise(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure A [M, K], B [K, N]
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [{M}, {K}] and B is [{Kb}, {N}]"
    # Make inputs contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Output in fp32 for stability
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Launch grid: one program per row and per column tile
    grid = (M, triton.cdiv(N, 128))  # initial grid; autotune configs control BLOCK_N
    matmul_rowwise_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton implementation: C = A @ B.T
        return triton_matmul_rowwise(A, B)


def run(*args):
    return ModelNew()(*args)
