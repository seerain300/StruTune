import torch
import triton
import triton.language as tl


# General 2D matmul: C = A @ BT, where A: [M, K], BT: [K, N], C: [M, N]
@triton.jit
def matmul_at_bT_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        # Pointers for BT tile [BLOCK_K, BLOCK_N]
        bt_ptrs = BT + offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn

        # 1D masks for rows and cols, then broadcast
        row_mask = offs_m < M          # shape: (BLOCK_M,)
        col_mask = offs_n < N          # shape: (BLOCK_N,)
        k_mask = offs_k < K            # shape: (BLOCK_K,)

        # Build 2D masks by broadcasting
        a_mask = row_mask[:, None] & k_mask[None, :]      # (BLOCK_M, BLOCK_K)
        b_mask = k_mask[:, None] & col_mask[None, :]      # (BLOCK_K, BLOCK_N)

        # Load tiles with masks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        bt = tl.load(bt_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results for this tile
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


# Specialized kernel for M == 1: computes C[0, :] = A[0, :] @ B.T
@triton.jit
def row_matmul_kernel(
    A_row, BT, C_row,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Single program computes the entire output row (vector length N)
    offs_n = tl.arange(0, BLOCK_N)  # vector across N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float16)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_row[k] as a vector
        a_vec = tl.load(A_row + offs_k * stride_am, mask=offs_k < K, other=0.0)  # (BLOCK_K,)

        # Load BT[k, offs_n] as a tile (BLOCK_K, BLOCK_N)
        bt_ptrs = BT + offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt_tile = tl.load(bt_ptrs, mask=bt_mask, other=0.0)  # (BLOCK_K, BLOCK_N)

        # Compute partial dot for this chunk: sum over K
        # Broadcast a_vec to (BLOCK_K, BLOCK_N) and sum along axis 0
        partial = tl.sum(bt_tile * a_vec[:, None], axis=0)  # (BLOCK_N,)
        acc += partial

    # Store the resulting row
    c_ptrs = C_row + offs_n * stride_cm
    row_mask = offs_n < N
    tl.store(c_ptrs, acc, mask=row_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguity
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, "B's second dimension must match A's second dimension"

        # BT = B.T with shape [K, N]
        BT = B.T.contiguous()

        # Output tensor C [M, N] in float16 to match inputs
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Tile sizes: conservative and commonly supported
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Launch general kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        return C


def run(*args):
    return ModelNew()(*args)
