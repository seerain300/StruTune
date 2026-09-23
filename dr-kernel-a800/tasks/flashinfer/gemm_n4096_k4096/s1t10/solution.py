import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_init

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Compute pointers
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_M, BLOCK_K], fp16/fp32
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N], fp16/fp32

        # Accumulate
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def rowwise_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Each program handles one row m and a tile of columns
    pid_m = tl.program_id(0)  # row index
    pid_n = tl.program_id(1)  # column tile index

    m = pid_m
    # Compute column offsets for this tile
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Bounds masks
    m_in_bounds = m < M
    n_mask = offs_n < N

    # Accumulator for this row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K

        # Load A_row_slice [BLOCK_K]
        a_row_ptrs = A_ptr + m * stride_am + k * stride_ak
        a_row = tl.load(a_row_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load B_T_block [BLOCK_K, BLOCK_N] via B's original layout B[n, k]
        b_block_ptrs = B_ptr + offs_n[None, :] * stride_bn + k[:, None] * stride_bk
        b_block = tl.load(b_block_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)  # [BLOCK_K, BLOCK_N]

        # acc += a_row[:, None] * b_block
        # tl.dot([1, BLOCK_K], [BLOCK_K, BLOCK_N]) -> [1, BLOCK_N], broadcast over rows
        partial = tl.dot(a_row[None, :], b_block)
        acc += partial[0, :]

    # Store result
    c_row_ptrs = C_ptr + m * stride_cm + offs_n * stride_cn
    tl.store(c_row_ptrs, acc, mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Expect A: [M, K], B: [N, K], output C: [M, N] = A @ B.T
        # We compute directly in Triton, no torch.matmul in forward.

        # Ensure contiguity
        A = A.contiguous()
        B = B.contiguous()

        M, K_a = A.shape
        N_b, K_b = B.shape
        assert K_a == K_b, f"Incompatible shapes: A is [M, {K_a}], B is [{N_b}, {K_b}]"
        K = K_a
        N = N_b

        # Output in fp32 for numerical stability, then cast to A.dtype
        # Strides (in elements)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)

        # Allocate fp32 output
        C_fp32 = torch.empty((M, N), device=A.device, dtype=torch.float32)
        stride_cm, stride_cn = C_fp32.stride(0), C_fp32.stride(1)

        # Choose kernel based on M
        if M <= 16:
            # Row-wise kernel: grid over (M rows, N tiles)
            BLOCK_N = 256
            BLOCK_K = 128
            grid = (M, triton.cdiv(N, BLOCK_N))
            rowwise_matmul_bt_kernel[grid](
                A, B, C_fp32,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3
            )
        else:
            # General 2D tiled kernel
            BLOCK_M = 128
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_bt_kernel[grid](
                A, B, C_fp32,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=3
            )

        # Cast to original A dtype for output (PyTorch typically keeps same dtype)
        return C_fp32.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
