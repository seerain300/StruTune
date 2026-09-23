import torch
import triton
import triton.language as tl

# Generic Triton GEMM: computes C[M, N] = A[M, K] @ B_T[K, N], where B_T[k, n] = B[n, k]
@triton.jit
def _matmul_generic_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,           # A strides: row-major [M, K]
    stride_bn, stride_bk,           # B strides: row-major [N, K] (we index as B_T[k, n] = B[n, k])
    stride_cm, stride_cn,           # C strides: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # columns of C/N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_range = k0 + tl.arange(0, BLOCK_K)
        # Load A[rm, k_range] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + k_range[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_range[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load B_T[k_range, cn] == B[cn, k_range] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + cn[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (cn[None, :] < N) & (k_range[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate outer products: acc += a @ b
        acc += tl.dot(a, b)  # a: [BM, BK], b: [BK, BN] -> [BM, BN]

        k0 += BLOCK_K

    # Store results
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Optional: a specialized kernel for M == 1 can be added if desired, but we keep a single robust kernel here.
# The forward will pass M, N, K, and strides and Triton will handle the M==1 case via BLOCK_M=1 and grid (1, N_tiles).

class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure dtype is float16 for the input tensors
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        M, K = A.shape
        N = B.shape[0]  # B is [N, K]; C = A @ B.T -> output [M, N]

        # Output buffer in float32 for accumulation
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose tile sizes; tuned for N=4096, K=4096 but work for general sizes
        BLOCK_M = 1 if M == 1 else 64
        BLOCK_N = 256
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_generic_at_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        # Cast back to original A dtype to match torch.matmul behavior
        return C.to(A.dtype)