import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,  # BT has shape [K, N], strides (stride_bTk, stride_bTn)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk  # current K indices in this chunk

        # Pointers for A[m, k]
        a_ptrs = A + rm[:, None] * stride_am + k[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Pointers for B_T[k, n] -> BT has shape [K, N]
        b_ptrs = BT + k[:, None] * stride_bTk + rn[None, :] * stride_bTn
        b_mask = (k[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)  # A: [M, K], B_T: [K, N] => [M, N]

    # Store result
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [N, K]
        # We need C = A @ B.T with shape [M, N]
        M, K = A.shape
        N = B.shape[0]

        # Explicitly create B_T with correct shape and strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Allocate output in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose tile sizes; masks handle edges
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_at_bt_kernel[grid](
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


def run(*args):
    return ModelNew()(*args)
