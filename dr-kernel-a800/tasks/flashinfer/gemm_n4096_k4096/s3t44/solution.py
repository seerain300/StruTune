import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bt, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K], A is (M, K)
        A_ptrs = A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # BT tile: BT is (K, N) since BT = B.T with B of shape (N, K)
        BT_ptrs = BT + k_offsets[:, None] * stride_bt + n_offsets[None, :] * stride_bn

        # Masks for bounds
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load with masking, zeros for out-of-bounds
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        BT_tile = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate: A_tile [BM, BK] dot BT_tile [BK, BN] -> [BM, BN]
        acc += tl.dot(A_tile.to(tl.float32), BT_tile.to(tl.float32))

    # Store result with masking
    C_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "Triton kernels require CUDA tensors"
        # Shapes: A (M, K), B (N, K), C (M, N)
        M, K_A = A.shape
        N, K_B = B.shape
        assert K_A == K_B, "A's second dimension must equal B's first dimension"
        # Prepare BT = B.T with shape (K, N)
        BT = B.transpose(0, 1).contiguous()
        # Output tensor
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bt = BT.stride(0)  # along K
        stride_bn = BT.stride(1)  # along N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling and grid
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K_A,
            stride_am, stride_ak,
            stride_bt, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=4,
        )
        return C


def run(*args):
    return ModelNew()(*args)
