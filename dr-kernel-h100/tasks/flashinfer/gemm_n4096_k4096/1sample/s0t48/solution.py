import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_matmul_1d_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # This kernel is specialized for M == 1. It accumulates over K and writes across N.
    # We assume M == 1; otherwise, the host should not call this kernel.
    # Each program handles a block of columns [cols_start, cols_start + BLOCK_N).
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    # Initialize accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    # Note: We use a simple for k in range(0, K, BLOCK_K) loop. Triton will compile this pattern.
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Compute pointers for A[0, k_offsets] (vector of length BLOCK_K)
        A_ptrs = A + 0 * stride_am + k_offsets * stride_ak  # A[m, k] with m=0
        a_vec = tl.load(A_ptrs, mask=mask_k, other=0.0)

        # Compute pointers for BT[k_offsets, cols] (tile: BLOCK_K x BLOCK_N)
        BT_ptrs = BT + k_offsets[:, None] * stride_bTk + cols[None, :] * stride_bTn
        bt_tile = tl.load(BT_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Elementwise multiply and reduce over K (sum across rows of bt_tile)
        # acc += sum_k a_vec[k] * bt_tile[k, :]
        # Implement reduction over BLOCK_K using tl.sum on axis 0
        acc += tl.sum(a_vec[:, None] * bt_tile, axis=0)

    # Store results to C[0, cols]
    C_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(C_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        """
        Computes C = A @ B.T
        A: [M, K], B: [N, K], returns C: [M, N]
        Triton is used for the common M==1 case. For other M, we fall back to torch.matmul(B.T).
        """
        M, K = A.shape
        N = B.shape[1]  # since B is [N, K]

        # Handle M == 1 with Triton kernel
        if M == 1:
            # Explicitly transpose B to [K, N] for predictable strides
            BT = B.transpose(0, 1).contiguous()  # BT: [K, N]
            # Output buffer (row 0, columns N)
            C = torch.empty((1, N), device=A.device, dtype=A.dtype)

            # Choose block sizes; tune for performance. For N=4096, K=4096, these are reasonable defaults.
            BLOCK_N = 256  # columns per program
            BLOCK_K = 256  # reduction chunk over K

            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_matmul_1d_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                num_warps=4, num_stages=3,
            )

            return C

        # Fallback for M > 1: use robust PyTorch matmul
        BT = B.transpose(0, 1)  # no need to make contiguous; PyTorch matmul handles strides
        C = A @ BT  # A: [M, K], BT: [K, N] -> C: [M, N]
        return C


def run(*args):
    return ModelNew()(*args)
