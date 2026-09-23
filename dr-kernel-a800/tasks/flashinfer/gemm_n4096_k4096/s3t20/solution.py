import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, Out_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_k, BT_stride_n,     # BT is (K, N), contiguous: stride_k = N, stride_n = 1
    Out_stride_m, Out_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BK]

        # Load A chunk: A[rows, k], shape [BM, BK]
        a_ptrs = A_ptr + rows[:, None] * A_stride_m + k[None, :] * A_stride_k
        a_mask = (rows[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # fp16

        # Load BT chunk: BT[k, cols], shape [BK, BN] (since BT is (K, N), contiguous)
        bt_ptrs = BT_ptr + k[:, None] * BT_stride_k + cols[None, :] * BT_stride_n
        bt_mask = (k[:, None] < K) & (cols[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)  # fp16

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Store results to Out with proper masking
    out_ptrs = Out_ptr + rows[:, None] * Out_stride_m + cols[None, :] * Out_stride_n
    out_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)  # acc is fp32; will cast on store if Out is fp16


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (M, K), B: (N, K), output C: (M, N) = A @ B.T
        assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D"
        M, K = A.shape
        N_B, K_B = B.shape
        assert K == K_B, "Inner dimension K must match"
        N = N_B

        # Ensure inputs are contiguous for predictable strides
        A = A.contiguous()
        # Compute BT as contiguous to guarantee correct strides for the kernel
        BT = B.T.contiguous()

        # Output tensor in fp16 (to match original run's dtype)
        out = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Heuristics for tile sizes and kernel selection
        if M >= 128 and N >= 128:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
            num_warps, num_stages = 8, 4
            # Launch Triton kernel
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_at_bT_kernel[grid](
                A, BT, out,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
            return out
        elif M >= 64 and N >= 128:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
            num_warps, num_stages = 8, 3
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_at_bT_kernel[grid](
                A, BT, out,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
            return out
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            num_warps, num_stages = 4, 3
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_at_bT_kernel[grid](
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
