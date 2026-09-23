import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_at_bT_kernel_fp16(
    A, BT, C,
    M, N, K,
    A_stride0, A_stride1,
    BT_stride0, BT_stride1,
    C_stride0, C_stride1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # 2D grid of programs; each computes a tile of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: (BLOCK_M, BLOCK_K), A is (M, K)
        a_ptrs = A + (offs_m[:, None] * A_stride0 + offs_k[None, :] * A_stride1)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # fp16

        # Load BT tile: BT is (K, N); load BT[offs_k, offs_n]
        bt_ptrs = BT + (offs_k[:, None] * BT_stride0 + offs_n[None, :] * BT_stride1)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)  # fp16

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Store results to C with masking
    c_ptrs = C + (offs_m[:, None] * C_stride0 + offs_n[None, :] * C_stride1)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


@triton.jit
def _row_matvec_kernel_fp16(
    A_row, BT, C_row,
    M, N, K,
    A_row_stride0, A_row_stride1,
    BT_stride0, BT_stride1,
    C_row_stride0, C_row_stride1,
    BLOCK_K: tl.constexpr,
):
    # Compute C_row[0, :] = A_row[0, :] @ BT[:, :]
    # Only one row (M == 1), so we launch a single CTA
    acc = tl.zeros((N,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        # Load A_row[0, k]
        a_ptrs = A_row + (0 * A_row_stride0 + k * A_row_stride1)
        a_mask = k < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # fp16

        # Load BT[k, offs_n]
        offs_n = tl.arange(0, N)
        bt_ptrs = BT + (k[:, None] * BT_stride0 + offs_n[None, :] * BT_stride1)
        bt_mask = (k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)  # fp16

        a32 = a.to(tl.float32)
        bt32 = bt.to(tl.float32)
        # Multiply row-wise and reduce along k
        prod = a32[:, None] * bt32  # (BLOCK_K, N)
        acc += tl.sum(prod, axis=0)

    # Store result to C_row[0, :]
    c_ptrs = C_row + (0 * C_row_stride0 + tl.arange(0, N) * C_row_stride1)
    c_mask = tl.arange(0, N) < N
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA device
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        M, K = A.shape
        N, KB = B.shape
        assert KB == K, "B must have shape (N, K) matching A's second dimension."

        # Make B.T contiguous as (K, N)
        BT = B.transpose(0, 1).contiguous()  # shape (K, N)

        # Output tensor
        out = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Specialized path for M == 1: compute single output row
        if M == 1:
            A_row = A.contiguous()  # (1, K)
            C_row = torch.empty((1, N), dtype=torch.float16, device=A.device)
            grid = (1, 1)
            _row_matvec_kernel_fp16[grid](
                A_row, BT, C_row,
                M, N, K,
                A_row.stride(0), A_row.stride(1),
                BT.stride(0), BT.stride(1),
                C_row.stride(0), C_row.stride(1),
                BLOCK_K=128,
                num_warps=4, num_stages=3,
            )
            return C_row  # shape (1, N)

        # General 2D tiling kernel
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
        num_warps, num_stages = 8, 3
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_at_bT_kernel_fp16[grid](
            A, BT, out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
