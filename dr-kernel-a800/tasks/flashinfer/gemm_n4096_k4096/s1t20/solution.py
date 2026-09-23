import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced configs for general GEMM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        # Larger N tiles to reduce grid over N for big matrices
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
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
    # 2D tiling over C of shape [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A is [M, K], strides (stride_am, stride_ak)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B is [N, K], strides (stride_bn, stride_bk); emulate B^T via B[n, k] addressing
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def rowwise_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes one output row m
    m = tl.program_id(0)
    if m >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    acc_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A_row_slice [1, BLOCK_K]
        a_row = tl.load(A_ptr + m * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # shape [BLOCK_K]
        # Load B_T_block [BLOCK_K, BLOCK_N] using B[n, k] addressing
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b_block = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)  # shape [BLOCK_K, BLOCK_N]
        # Accumulate: outer product over BLOCK_K
        for kk in range(BLOCK_K):
            a_k = a_row[kk]  # scalar fp32
            b_k = b_block[kk, :]  # vector [BLOCK_N], fp32
            acc_vec += a_k * b_k

    # Store the computed row to C
    c_ptrs = C_ptr + m * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc_vec, mask=offs_n < N)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N_b, K_b = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is [{M}, {K}], B is [{N_b}, {K_b}]; need K==K_b")

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Output in fp32 for numerical stability; cast to A.dtype before returning
        C_fp32 = torch.empty((M, N_b), dtype=torch.float32, device=A.device)

        # Strides
        stride_am, stride_ak = A.stride()
        stride_bn, stride_bk = B.stride()
        stride_cm, stride_cn = C_fp32.stride()

        # If M is small, use the row-wise kernel to minimize overhead
        if M <= 64:
            grid = (M,)  # one program per row
            rowwise_matmul_bt_kernel[grid](
                A, B, C_fp32,
                M, N_b, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            # General GEMM tiling over M and N
            grid = (triton.cdiv(M, 128), triton.cdiv(N_b, 128))
            matmul_bt_kernel[grid](
                A, B, C_fp32,
                M, N_b, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )

        return C_fp32.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
