import torch
import triton
import triton.language as tl

@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D program ids for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile: B[k, n]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result to C[m, n] as fp16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and fp16 dtype for Triton execution
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}).")

        # Contiguous inputs for robust stride handling
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output tensor, contiguous, fp16
        C = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        # Launch Triton GEMM with a 2D grid
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _gemm_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return C


def run(*args):
    return ModelNew()(*args)
