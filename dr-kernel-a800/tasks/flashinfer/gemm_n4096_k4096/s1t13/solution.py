import torch
import triton
import triton.language as tl


@triton.jit
def rowwise_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # A strides: (row, col)
    stride_bn, stride_bk,     # B strides: (row, col) for B[n, k]
    BLOCK_N: tl.constexpr,    # columns per chunk
    BLOCK_K: tl.constexpr,    # reduction per chunk
):
    # Each program instance computes one output row 'm' and iterates over N and K
    m = tl.program_id(0)
    if m >= M:
        return

    # Loop over columns in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Reduction over K in chunks of BLOCK_K
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

            # Load A[m, k_offsets] -> [BLOCK_K]
            a_row_base = A_ptr + m * stride_am
            a_ptrs = a_row_base + k_offsets * stride_ak  # [BLOCK_K]
            a_vec = tl.load(a_ptrs, mask=k_offsets < K, other=0.0)  # fp32

            # Load B[offs_n, k_offsets] -> [BLOCK_N, BLOCK_K]
            b_ptrs = B_ptr + offs_n[:, None] * stride_bn + k_offsets[None, :] * stride_bk
            b_mask = (offs_n[:, None] < N) & (k_offsets[None, :] < K)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)  # fp32

            # Accumulate: acc += sum over k of a_vec[k] * b_block[:, k]
            for kk in range(BLOCK_K):
                b_col_kk = b_block[:, kk]  # [BLOCK_N]
                acc += a_vec[kk] * b_col_kk

        # Store the computed chunk to C[m, n_start:n_start+BLOCK_N]
        c_ptrs = C_ptr + m * N + offs_n  # C is [M, N] with row stride N
        tl.store(c_ptrs, acc, mask=offs_n < N)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D tiling over output rows (M) and columns (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile as B.T: B_T[offs_k, offs_n] = B[offs_n, offs_k]
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # acc += A_tile @ B_tile (reduce over K dimension), accumulate in fp32
        acc += tl.dot(a_tile, b_tile)

    # Store acc to C[offs_m, offs_n]
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure inputs are CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N = B.shape[1]  # B is [N, K]

        # Allocate fp32 output for numerical stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose kernel based on M
        if M <= 16:
            # Row-wise kernel: one program per output row
            grid = (M,)
            rowwise_mm_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                BLOCK_N=256,
                BLOCK_K=128,
                num_warps=4,
            )
        else:
            # 2D tiled kernel for larger M, autotuned
            # Grid over output tiles
            # We set BLOCK_M and BLOCK_N via the autotuned configs internally
            # We need to provide grid based on a default tile to compute launch size;
            # Triton will choose the config at runtime. We can pick an upper bound for grid.
            # Use conservative grid based on largest typical BLOCK_M/BLOCK_N in configs.
            # Alternatively, compute grid with a default tile (e.g., 64x128).
            BLOCK_M = 64
            BLOCK_N = 128
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
            )

        # Cast output to A's dtype to match typical behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
