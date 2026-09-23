import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel_fp16(
    A, BT, C,
    M, N, K,
    A_stride0, A_stride1,
    BT_stride0, BT_stride1,
    C_stride0, C_stride1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N; loop over K in chunks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A + (offs_m[:, None] * A_stride0) + (offs_k[None, :] * A_stride1)
        # Pointers for BT tile: BT is (K, N); we index BT[offs_k, offs_n]
        bt_ptrs = BT + (offs_k[:, None] * BT_stride0) + (offs_n[None, :] * BT_stride1)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)       # (BLOCK_M, BLOCK_K), fp16
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)    # (BLOCK_K, BLOCK_N), fp16

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))  # (BLOCK_M, BLOCK_N)

    # Store results to C with mask
    c_ptrs = C + (offs_m[:, None] * C_stride0) + (offs_n[None, :] * C_stride1)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


@triton.jit
def row_matvec_kernel_fp16(
    A_row, BT, C_row,
    M, N, K,
    A_row_stride0, A_row_stride1,
    BT_stride0, BT_stride1,
    C_row_stride0, C_row_stride1,
    BLOCK_K: tl.constexpr,
):
    # Compute C_row[0, :] = A_row[0, :] @ BT[:, :]
    # Only one row (M == 1), so we launch a single CTA
    # Output vector length N
    acc = tl.zeros((N,), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load scalar A_row[0, k] (M==1 => row index is 0)
        a_ptrs = A_row + (0 * A_row_stride0) + (offs_k * A_row_stride1)
        a = tl.load(a_ptrs, mask=offs_k < K, other=0.0)  # (BLOCK_K,), fp16
        # Load BT[:, k] as a vector of length BLOCK_K, then we need to expand a to (1, BLOCK_K) to form outer product
        bt_ptrs = BT + (offs_k * BT_stride0) + (tl.arange(0, BLOCK_K) * BT_stride1)
        bt = tl.load(bt_ptrs, mask=offs_k < K, other=0.0)  # (BLOCK_K,), fp16
        # Outer product contribution: sum over k of a[k] * bt[k]
        # To form outer product, we need a as (1, BLOCK_K) and bt as (BLOCK_K, 1)
        a_vec = a[:, None]  # (1, BLOCK_K)
        bt_vec = bt[None, :]  # (1, BLOCK_K) horizontal? We need (BLOCK_K, 1).
        # Create a (BLOCK_K, 1) by transposing bt? Triton expects broadcasting via [:, None]. Instead, compute directly:
        # For each k, contribute a[k] * bt[k] to acc
        # Implement as a simple loop over BLOCK_K (small, acceptable):
        for kk in range(BLOCK_K):
            # Only update if offs_k < K
            if (k0 + kk) < K:
                acc += a[kk] * bt[kk]

    # Store result vector to C_row[0, :]
    c_ptrs = C_row + (0 * C_row_stride0) + (tl.arange(0, N) * C_row_stride1)
    c_mask = tl.arange(0, N) < N
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton
        assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA device for Triton."
        # Shapes
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"Incompatible shapes: A is (M, K), B is (N, K) expected."

        # Output tensor, match dtype of original (float16)
        out = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Compute BT = B.T contiguous to simplify indexing in kernel
        BT = B.transpose(0, 1).contiguous()

        # Choose tile sizes dynamically based on shape to improve performance
        if M <= 32:
            BLOCK_M = 32
            num_warps = 4
        elif M <= 64:
            BLOCK_M = 64
            num_warps = 4
        else:
            BLOCK_M = 128
            num_warps = 8

        BLOCK_N = 128
        BLOCK_K = 32
        num_stages = 3

        # Grid for 2D tiling
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        if M == 1:
            # Specialized single-row kernel: compute C[0, :]
            # C_row is a 1xN tensor; allocate with empty_like to keep dtype and device
            C_row = torch.empty((1, N), device=A.device, dtype=torch.float16)
            # Strides
            A_row_stride0, A_row_stride1 = A.stride(0), A.stride(1)
            BT_stride0, BT_stride1 = BT.stride(0), BT.stride(1)
            C_row_stride0, C_row_stride1 = C_row.stride(0), C_row.stride(1)
            row_matvec_kernel_fp16[(1,)](
                A[0, :], BT, C_row,
                M, N, K,
                A_row_stride0, A_row_stride1,
                BT_stride0, BT_stride1,
                C_row_stride0, C_row_stride1,
                BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
            # Write C_row to out[0, :]
            out[0, :] = C_row[0, :]
        else:
            # General 2D kernel
            matmul_at_bT_kernel_fp16[grid](
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
