import torch
import triton
import triton.language as tl

# Triton kernel: computes C = A @ B_T, where B_T is the transpose of B (logical).
# A: [M, K], B: [N, K] (original B), we index as B_T[k, n] = B[n, k] using strides.
# C: [M, N]
@triton.jit
def _matmul_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: row (m), col (k)
    stride_bn, stride_bk,   # B strides: row (n), col (k)  -> we will index as B_T[k, n] = B[n, k]
    stride_cm, stride_cn,   # C strides: row (m), col (n)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the current tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape [BLOCK_N]

    # Create accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # shape [BLOCK_K]

        # Pointers for A block: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak

        # Pointers for B^T block: we want B_T[k, n] = B[n, k]
        # So for each (k, n), address = B_ptr + n * stride_bn + k * stride_bk
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk

        # Masks for bounds
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)

        # Load tiles
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))

    # Store results
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A @ B.T using Triton. Ensures correctness for general shapes.
        A: [M, K], B: [N, K] (original B). We index B as B_T[k, n] = B[n, k].
        Returns C: [M, N] with dtype = A.dtype.
        """
        # Ensure tensors are on CUDA and contiguous for best performance
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N = B.shape[0]  # B is [N, K], N is first dim

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)  # accumulate in fp32

        # Choose tile sizes. These are decent defaults and divide common sizes (e.g., 4096).
        # For M=1, we still use BLOCK_M=1; grid will be (1, N_tiles).
        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
        _matmul_at_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast to original dtype to match typical output behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
