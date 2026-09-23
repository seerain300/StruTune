import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # strides for A: A[M, K]
    stride_btk, stride_btn,     # strides for BT: BT[K, N] (B transposed)
    stride_cm, stride_cn,       # strides for C: C[M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D program id
    pid_m = tl.program_id(0)  # tile index over M
    pid_n = tl.program_id(1)  # tile index over N

    # Offsets for rows and columns handled by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Compute pointers for current tiles
        # A: [M, K] -> row offs_m, K-dim offs_k
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # BT: [K, N] -> K-dim offs_k, N-dim offs_n
        b_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)

        # Build 1D masks for rows, cols, and K range
        row_mask_1d = offs_m < M            # [BLOCK_M]
        col_mask_1d = offs_n < N            # [BLOCK_N]
        k_mask_1d = offs_k < K              # [BLOCK_K]

        # Broadcast to 2D masks
        a_mask = row_mask_1d[:, None] & k_mask_1d[None, :]  # [BLOCK_M, BLOCK_K]
        b_mask = k_mask_1d[:, None] & col_mask_1d[None, :]  # [BLOCK_K, BLOCK_N]

        # Load tiles (masked), 'other=0' for out-of-bounds
        a = tl.load(a_ptrs, mask=a_mask, other=0)  # dtype inferred from pointer; here float16 as A is float16
        b = tl.load(b_ptrs, mask=b_mask, other=0)

        # Accumulate: elementwise multiply and reduce over K
        # Using tl.dot on (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, b)

    # Store results to C[M, N]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # Store mask only about M and N bounds (K handled in loop)
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure CUDA tensors for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        # Prepare BT = B.T as contiguous [K, N]
        BT = B.t().contiguous()
        M, K = A.shape
        N = BT.shape[1]  # since BT is (K, N)
        # Allocate output tensor
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Tile sizes chosen to balance performance and resource usage
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel (always; no torch fallback in Triton path)
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C


def run(*args):
    return ModelNew()(*args)
