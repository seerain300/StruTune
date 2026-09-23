import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_at_btransposed_kernel(
    A, B_T, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[m, k]
        a_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_M, BLOCK_K]

        # Pointers for B_T[k, n] which is a contiguous transposed view: strides (K, 1)
        b_ptrs = B_T + k_offsets[:, None] * stride_bTk + n_offsets[None, :] * stride_bTn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate outer products in fp32
        acc += tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32))

    # Write back results
    c_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on the same device and have correct dtype
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
        assert A.dtype in (torch.float16, torch.float32), "A must be float16 or float32."
        assert B.dtype in (torch.float16, torch.float32), "B must be float16 or float32."

        # Compute B.T as a view (no data copy)
        BT = B.transpose(0, 1)  # BT: [K, N] view
        M, K = A.shape
        N = B.shape[0]

        # Output in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose tile sizes; masks handle edges
        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_at_btransposed_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast to original dtype to match torch.matmul behavior
        return C.to(A.dtype)