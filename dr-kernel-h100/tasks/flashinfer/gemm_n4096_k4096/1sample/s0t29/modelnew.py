import torch
import triton
import triton.language as tl

# General 2D kernel: computes C = A @ B_T, where B_T[k, n] = B[n, k].
@triton.jit
def _matmul_at_bt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # B strides for [N, K] -> B_T[k, n] = B[n, k]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A and C
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C
    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)  # k indices

        # Pointers for A[m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype follows A (e.g., fp16), cast to fp32 for acc

        # Pointers for B_T[k, n] = B[n, k] -> shape [BLOCK_K, BLOCK_N]
        # Using B's strides: stride_bn for n, stride_bk for k
        b_ptrs = B + cn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_mask = (rk[:, None] < K) & (cn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype follows B, cast to fp32 for acc

        # Accumulate outer products: acc += a @ b^T
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        # Cast to fp32 for stable accumulation
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Write back to C (C is allocated with desired output dtype, e.g., fp16)
    c_ptrs = C + rm[:, None] * stride_cm + cn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (cn[None, :] < N)
    # Store fp32 acc; Triton will cast to C's dtype if needed
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on the same device
        assert A.device == B.device, "A and B must be on the same device"
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N = B.shape[0]  # B is [N, K], output C is [M, N]

        # Allocate output in the same dtype as A to match PyTorch matmul default behavior
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Choose tile sizes; tuned for common case but robust for general shapes
        BLOCK_M = 16
        BLOCK_N = 256
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_at_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )
        return C