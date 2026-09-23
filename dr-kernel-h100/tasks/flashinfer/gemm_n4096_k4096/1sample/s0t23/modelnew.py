import torch
import triton
import triton.language as tl

# Generic GEMM kernel: C = A @ B_T, where B_T[k, n] = B[n, k]
@triton.jit
def _matmul_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B_T strides: bn = B.stride(1), bk = B.stride(0)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices this program will handle
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A sub-tile: A[rm, k]
        a_ptrs = A + rm[:, None] * stride_am + k_range[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_range[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Pointers for B_T sub-tile: B_T[k, cn] = B[cn, k]
        # Using B's strides: B_ptr + cn * B.stride(1) + k * B.stride(0)
        b_ptrs = B + cn[None, :] * stride_bn + k_range[:, None] * stride_bk
        b_mask = (cn[None, :] < N) & (k_range[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Write back to C
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure dtypes for accumulation and output
        # Inputs are typically fp16; we compute in fp32 for stability.
        # We allocate C in fp32 and cast to A.dtype at the end to match original behavior.
        M, K = A.shape
        N = B.shape[0]  # B has shape [N, K]; C has shape [M, N]

        # Allocate output as fp32
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose reasonable tiles; masks handle edges
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_at_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(1), B.stride(0),  # B_T[k, n] = B[n, k]
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast to original dtype (usually float16) to match torch.matmul behavior
        return C.to(A.dtype)