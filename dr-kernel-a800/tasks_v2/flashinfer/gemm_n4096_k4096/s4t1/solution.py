import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile row/col indices
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C, index into A
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C, correspond to B^T's cols (n)

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A is contiguous: A[m, k] -> A_ptr + m * K + k
        a_ptrs = A_ptr + (m_offsets[:, None] * K + k_offsets[None, :])
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B is contiguous: B[n, k] -> B_ptr + n * N + k  (B^T[k, n] = B[n, k])
        b_ptrs = B_ptr + (n_offsets[None, :] * N + k_offsets[:, None])
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C as FP16 (C is allocated as FP16 and contiguous: C[m, n] -> C_ptr + m * N + n)
    c_ptrs = C_ptr + (m_offsets[:, None] * N + n_offsets[None, :])
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew expects exactly two input tensors: A and B.")
        A, B = args

        # Ensure both tensors are on the same device
        if A.device != B.device:
            raise ValueError("A and B must be on the same device.")

        # Make inputs contiguous for correct, simple indexing
        A = A.contiguous()
        B = B.contiguous()

        # Validate shapes: A [M, K], B [K, N]
        if A.dim() != 2:
            raise ValueError(f"A must be 2D [M, K], got shape {tuple(A.shape)}.")
        if B.dim() != 2:
            raise ValueError(f"B must be 2D [K, N], got shape {tuple(B.shape)}.")
        M, K = A.shape
        K_b, N = B.shape
        if K_b != K:
            raise ValueError(f"B must have shape [K, N]; got B.shape={tuple(B.shape)} with first dim {K_b} != K={K}.")

        # Output tensor: float16, contiguous
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Tiling parameters: robust defaults
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_transpose_kernel[grid](
            A, B, C,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
