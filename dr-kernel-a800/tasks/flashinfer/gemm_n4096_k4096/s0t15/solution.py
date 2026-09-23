import torch
import triton
import triton.language as tl

# Tiny-M kernel: 1D grid over N tiles; unroll loop over M (M is constexpr -> compile-time unrolling)
@triton.jit
def matmul_tinyM_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for each n in this tile across all M
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over M rows; Triton will unroll because M is constexpr
    for m in tl.static_range(M):
        # Address for A[m, :] and iterate K in chunks
        a_row_ptr = A_ptr + m * stride_am
        for k0 in range(0, K, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K
            # Load A[m, k] vector
            a = tl.load(
                a_row_ptr + k_offsets * stride_ak,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)  # [BLOCK_K], fp32
            # Load B_T[k, n] = B[n, k] for this N tile
            b = tl.load(
                B_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_N, BLOCK_K], fp32
            # Accumulate: acc[n] += sum_k a[k] * b[n, k]
            acc += tl.sum(a[None, :] * b, axis=1)

    # Store results for this M row
    tl.store(C_ptr + m * stride_cm + n_offsets * stride_cn, acc, mask=n_mask)


# General 2D kernel: tiled over M and N with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_general_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # Load B_T tile: [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b = tl.load(
            B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store
    tl.store(
        C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are CUDA tensors and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N = B.shape[0]  # B is [N, K] in the evaluator; we index B as B_T[k, n] = B[n, k]

        # Allocate output (fp16 to match original behavior)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(1)  # B's second dim (K)
        stride_bn = B.stride(0)  # B's first dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose kernel: tiny-M specialization when M is very small
        if M <= 16:
            # Grid over N tiles only; choose BLOCK sizes based on N to reduce masked work
            if N >= 8192:
                BLOCK_N = 512
                BLOCK_K = 32
                num_warps = 8
                num_stages = 2
            elif N >= 2048:
                BLOCK_N = 256
                BLOCK_K = 32
                num_warps = 4
                num_stages = 2
            else:
                BLOCK_N = 128
                BLOCK_K = 16
                num_warps = 4
                num_stages = 2

            grid = (triton.cdiv(N, BLOCK_N),)

            matmul_tinyM_kernel[grid](
                A, B, C,
                M=M, N=N, K=K,
                stride_am=stride_am, stride_ak=stride_ak,
                stride_bk=stride_bk, stride_bn=stride_bn,
                stride_cm=stride_cm, stride_cn=stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
        else:
            # General 2D grid over M and N
            # Use a heuristic grid; autotune will refine BLOCK sizes
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 256))
            matmul_general_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
