import torch
import triton
import triton.language as tl


# Row-wise Triton kernel: each program computes one output row across a tile of N.
# Specialized for very small M (e.g., M <= 8).
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
    ],
    key=["N", "K_T"],
)
@triton.jit
def matmul_rowwise_kernel(
    A_ptr,      # *fp16 [M, K_T]
    BT_ptr,     # *fp16 [K_T, N] (B.t().contiguous())
    C_ptr,      # *fp16 [M, N]
    M, N, K_T,  # sizes
    stride_am, stride_ak,    # A strides: (row, col)
    stride_btk, stride_btn,  # BT strides: (k, n) in contiguous BT
    stride_cm, stride_cn,    # C strides
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row index
    pid_n = tl.program_id(1)  # N tile index

    i = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this row across BLOCK_N columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K_T in chunks (BLOCK_K implicitly 128)
    for k0 in range(0, K_T, 128):
        offs_k = k0 + tl.arange(0, 128)
        # Load A[i, k:k+128] -> shape (1, 128)
        a_ptrs = A_ptr + i * stride_am + offs_k[None, :] * stride_ak
        a_vals = tl.load(a_ptrs, mask=offs_k[None, :] < K_T, other=0.0).to(tl.float32)  # (1, 128)
        # Load BT[k:k+128, n:n+BLOCK_N] -> shape (128, BLOCK_N)
        b_ptrs = BT_ptr + offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn
        b_vals = tl.load(b_ptrs, mask=(offs_k[:, None] < K_T) & (offs_n[None, :] < N), other=0.0).to(tl.float32)  # (128, BLOCK_N)
        # Accumulate over k-chunk
        acc += tl.sum(b_vals * a_vals, axis=0)  # (BLOCK_N,)

    # Store result row
    c_ptrs = C_ptr + i * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=offs_n < N)


# General 2D-tiled Triton kernel for matmul over MxN with K loop
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K_T"],
)
@triton.jit
def matmul_tiled_kernel(
    A_ptr,           # *fp16 [M, K_T]
    BT_ptr,          # *fp16 [K_T, N] (B.t().contiguous())
    C_ptr,           # *fp16 [M, N]
    M, N, K_T,       # sizes
    stride_am, stride_ak,     # strides for A: (row, col)
    stride_btk, stride_btn,   # strides for BT: (k, n) in contiguous BT
    stride_cm, stride_cn,     # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_T, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = BT_ptr + offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_T), other=0.0).to(tl.float32)  # (BM, BK)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K_T) & (offs_n[None, :] < N), other=0.0).to(tl.float32)  # (BK, BN)

        acc += tl.dot(a, b)  # (BM, BN)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"

        # Triton kernels expect fp16
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Contiguous for predictable strides
        A = A.contiguous()

        # Construct BT = B.T contiguous to avoid stride issues
        BT = B.t().contiguous()

        M, K_T = A.shape
        K_BT, N = BT.shape  # BT is [K_T, N]
        assert K_BT == K_T, "B.T's first dimension must match A's second dimension"

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_btk, stride_btn = BT.stride(0), BT.stride(1)  # BT contiguous: (N, 1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Choose kernel based on M
        if M <= 8:
            # Grid over rows and N tiles
            def grid_fn(meta):
                return (M, triton.cdiv(N, meta["BLOCK_N"]))
            matmul_rowwise_kernel[grid_fn](
                A, BT, C,
                M, N, K_T,
                stride_am, stride_ak,
                stride_btk, stride_btn,
                stride_cm, stride_cn,
            )
        else:
            def grid_fn(meta):
                return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
            matmul_tiled_kernel[grid_fn](
                A, BT, C,
                M, N, K_T,
                stride_am, stride_ak,
                stride_btk, stride_btn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
