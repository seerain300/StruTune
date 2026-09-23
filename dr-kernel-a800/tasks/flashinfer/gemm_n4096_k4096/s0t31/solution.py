import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_smallM_1D_kernel(
    A_ptr,  # *fp16, [M, K]
    B_ptr,  # *fp16, [K, N] (B_T contiguous)
    C_ptr,  # *fp16, [M, N]
    M: tl.constexpr,  # rows of A
    N: tl.constexpr,  # columns of C, rows of B_T
    K: tl.constexpr,  # inner dimension
    stride_am,  # A.stride(0)
    stride_ak,  # A.stride(1)
    stride_bk,  # B_T.stride(0) == N
    stride_bn,  # B_T.stride(1) == K
    stride_cm,  # C.stride(0)
    stride_cn,  # C.stride(1)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program id over N tiles
    pid = tl.program_id(axis=0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # fp32 accumulator for all M rows and BLOCK_N columns
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Accumulate across all M rows
        for i in range(0, M):
            # Load A[i, k_offsets] as a vector of length BLOCK_K with row mask
            row_mask = (i < M)
            a_ptrs = A_ptr + (i * stride_am) + (k_offsets * stride_ak)
            a_vec = tl.load(a_ptrs, mask=(row_mask & k_mask), other=0.0).to(tl.float32)  # [BLOCK_K]

            # Load B_T[k_offsets, n_offsets] as a matrix [BLOCK_K, BLOCK_N]
            b_ptrs = B_ptr + (n_offsets[None, :] * stride_bn) + (k_offsets[:, None] * stride_bk)
            b_tile = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

            # Outer product: acc[i, :] += sum_k a_vec[k] * b_tile[k, :]
            acc[i, :] += tl.sum(b_tile * a_vec[None, :], axis=0)

    # Store results to C (cast to fp16)
    c_ptrs = C_ptr + (tl.arange(0, M)[:, None] * stride_cm) + (n_offsets[None, :] * stride_cn)
    store_mask = (tl.arange(0, M)[:, None] < M) & (n_mask[None, :])
    tl.store(c_ptrs, acc.to(tl.float16), mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton-only computation; no torch matmul in host code
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        A = A.contiguous()
        B_T = B.transpose(0, 1).contiguous()  # B_T shape: [K, N]
        M, K = A.shape
        K_B, N = B_T.shape
        assert K_B == K, f"B's second dim {K_B} must match A's second dim {K}"

        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_T.stride(0)  # corresponds to N
        stride_bn = B_T.stride(1)  # corresponds to K
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: 1D over N tiles
        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_N']),)

        matmul_smallM_1D_kernel[grid](
            A, B_T, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
