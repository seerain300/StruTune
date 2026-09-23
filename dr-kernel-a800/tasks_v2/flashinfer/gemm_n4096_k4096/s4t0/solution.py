import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: shape [M, K]
    stride_bk, stride_bn,   # strides for B: shape [K, N]
    stride_cm, stride_cn,   # strides for C: shape [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows and columns of this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[m, k] tile (shape [BLOCK_M, BLOCK_K])
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Pointers for B_T[k, n] = B[n, k] tile (shape [BLOCK_K, BLOCK_N])
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks for valid elements
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles (inputs are FP16 in provided setup)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: a is [BM, BK], b is [BK, BN]
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store results to C as FP16
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew expects exactly two input tensors: A and B.")
        A, B = args

        # Ensure both on same device
        if A.device != B.device:
            raise ValueError("A and B must be on the same device.")

        # Make inputs contiguous for performance
        A = A.contiguous()
        B = B.contiguous()

        # Extract shapes
        if A.dim() != 2:
            raise ValueError(f"A must be 2D [M, K], got shape {tuple(A.shape)}.")
        if B.dim() != 2:
            raise ValueError(f"B must be 2D [K, N], got shape {tuple(B.shape)}.")
        M, K = A.shape
        K_b, N = B.shape
        if K_b != K:
            raise ValueError(f"B must have shape [K, N]; got B.shape={tuple(B.shape)} with first dim {K_b} != K={K}.")

        # Output tensor: match original behavior, dtype=float16
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
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
