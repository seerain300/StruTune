import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile identifiers across M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute indices this program handles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator (keep in fp32 for numerical stability)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A and B tiles
        # A: [M, K] with strides (stride_am, stride_ak)
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B: [K, N] with strides (stride_bk, stride_bn); accessing B[k, n] (i.e., B.T)
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks to guard OOB
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (inputs are fp16; Triton load returns fp16)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: for BLOCK_M=1, this is per-row accumulation
        acc += tl.dot(A_tile, B_tile)

    # Store result
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("A and B must be 2D tensors")
        # Make inputs contiguous for simple stride handling
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2,


def run(*args):
    return ModelNew()(*args)
