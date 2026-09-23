import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_1d_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1D grid over N tiles
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for a row block (M x BLOCK_N)
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[m, k_offsets] for m in 0..M-1 -> shape (M, BLOCK_K)
        a_ptrs = A_ptr + m * stride_am + k_offsets[None, :] * stride_ak
        # We need to build a 2D pointer array for A[m, k] across m and k
        # Using a Python list comprehension to create a tuple of pointers for each m
        a = []
        for m_idx in range(0, M):
            a.append(tl.load(a_ptrs.replace('m', m_idx), mask=k_mask, other=0.0))
        a = tl.stack(a, axis=0)  # shape (M, BLOCK_K)

        # Load B_T[k_offsets, n_offsets] = B[n_offsets, k_offsets], shape (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate: (M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (M, BLOCK_N)
        acc += tl.dot(a, b)

    # Store results
    c_ptrs = C_ptr + m * stride_cm + n_offsets[None, :] * stride_cn
    # Store with mask for m (all m valid) and n (boundary)
    tl.store(c_ptrs, acc, mask=n_mask[None, :])


# Optional general 2D kernel (kept for completeness, but 1D kernel should be faster for tiny M)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
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

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        a = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )  # (BLOCK_M, BLOCK_K)
        b = tl.load(
            B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0
        )  # (BLOCK_K, BLOCK_N)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors. No torch matmul or PyTorch computation here.
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"B's second dim ({K2}) must match A's second dim ({K})."

        # Output tensor in fp16 to match original behavior
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k]
        stride_bn = B.stride(0)  # original first dim (N)
        stride_bk = B.stride(1)  # original second dim (K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Prefer 1D kernel for tiny M to maximize parallelism along N
        def grid_1d(meta):
            return (triton.cdiv(N, meta['BLOCK_N']),)

        matmul_1d_kernel[grid_1d](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
