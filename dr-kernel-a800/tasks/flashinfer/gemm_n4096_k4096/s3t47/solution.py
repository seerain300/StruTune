import torch
import triton
import triton.language as tl

# Triton kernel: C = A @ BT, where BT = B.T with shape [K, N]
# A: [M, K], BT: [K, N], C: [M, N]
@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_k, BT_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A[m, k] and BT[k, n]
        a_ptrs = A_ptr + (m_offsets[:, None] * A_stride_m) + (k_offsets[None, :] * A_stride_k)
        bt_ptrs = BT_ptr + (k_offsets[:, None] * BT_stride_k) + (n_offsets[None, :] * BT_stride_n)

        # Boundary masks
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles and cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Outer-product accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(a, bt)

    # Store results to C[m, n] with mask
    c_ptrs = C_ptr + (m_offsets[:, None] * C_stride_m) + (n_offsets[None, :] * C_stride_n)
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)  # Triton will cast to the pointer dtype as needed


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA device and shapes
        assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA for Triton kernel."
        M, K_A = A.shape
        N, K_B = B.shape
        assert K_A == K_B, f"Incompatible shapes: A is (M, {K_A}) and B is ({N}, {K_B}). Expected K to match."

        # Make inputs contiguous and create BT = B.T contiguous
        A_contig = A.contiguous()
        BT = B.t().contiguous()  # BT: [K, N]

        # Allocate output tensor (same dtype as inputs)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Tile sizes tuned for performance without exceeding shared memory
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A_contig, BT, C,
            M, N, K_A,
            A_contig.stride(0), A_contig.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        return C


def run(*args):
    return ModelNew()(*args)
