import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # note: stride_bn = B.stride(1), stride_bk = B.stride(0) for B_T indexing
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single row (M == 1), process columns in tiles of BLOCK_N
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    # Output row is 0 (since M==1), accumulate in fp32
    out = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        # Mask for valid k_range
        k_mask = k_range < K

        # Load A[0, k_range]: M==1 so rm=0
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K], fp16

        # Load B_T[k_range, cols] as B[cols, k_range] using B strides:
        # B_T[k, n] = B[n, k] -> address = B_ptr + n * B.stride(1) + k * B.stride(0)
        b_ptrs = B + cols[None, :] * stride_bk + k_range[:, None] * stride_bn
        b_mask = (k_mask[:, None]) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N], fp16

        # Accumulate outer product: out += sum_k a[k] * b[k, :]
        # Cast to fp32 for accumulation
        a_fp32 = a.to(tl.float32)            # [BLOCK_K]
        b_fp32 = b.to(tl.float32)            # [BLOCK_K, BLOCK_N]
        out += tl.sum(a_fp32[:, None] * b_fp32, axis=0)

    # Store results to C[0, cols]
    C_ptrs = C + 0 * stride_cm + cols * stride_cn
    # Cast back to original dtype (fp16) before storing
    tl.store(C_ptrs, out.to(tl.float16), mask=cols < N)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B_T[k, n] = B[n, k] -> use B.stride(1) and B.stride(0)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N, reduce over K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # B_T tile: [BLOCK_K, BLOCK_N], load as B[n, k] with strides (bk, bn)
        b_ptrs = B + rn[None, :] * stride_bk + rk[:, None] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # FMA: acc += A_tile @ B_tile.T
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Store C tile
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Output shape: [M, N], with A: [M, K], B: [N, K], C = A @ B.T => [M, N]
        M, K = A.shape
        N = B.shape[0]
        # We'll compute in fp32 for stability and cast to fp16 at the end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Fast path specialized for M == 1: focus on columns of C
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),  # critical: B_T[k, n] = B[n, k]
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic 2D GEMM for other M
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),  # B_T indexing via B strides
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul default behavior (fp16 inputs)
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
