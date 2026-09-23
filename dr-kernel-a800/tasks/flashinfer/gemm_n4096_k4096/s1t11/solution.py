import torch
import triton
import triton.language as tl


@triton.jit
def matmul_rowwise_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,      # A strides: row (m), col (k)
    stride_bn, stride_bk,      # B strides: row (n), col (k)
    stride_cm, stride_cn,      # C strides: row (m), col (n)
    BLOCK_N: tl.constexpr,     # number of columns processed per iteration
    BLOCK_K: tl.constexpr      # reduction chunk over K
):
    # Each program instance computes one output row m
    m = tl.program_id(0)
    if m >= M:
        return

    # Base pointer for this row in A
    a_row_base = A_ptr + m * stride_am

    n_start = 0
    while n_start < N:
        offs_n = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        n_mask = offs_n < N

        # Accumulator for this row over the BLOCK_N columns
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        k_start = 0
        while k_start < K:
            k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = k_offsets < K

            # Load A row slice: A[m, k_offsets] -> [BLOCK_K]
            a_ptrs = a_row_base + k_offsets * stride_ak
            a_vec = tl.load(a_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K], fp32 promotion

            # Load B block: B[offs_n, k_offsets], shape [BLOCK_N, BLOCK_K]
            b_ptrs = B_ptr + offs_n[:, None] * stride_bn + k_offsets[None, :] * stride_bk
            b_block = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_K], fp32

            # Vectorized reduction over K: a_vec is [BLOCK_K], b_block is [BLOCK_K, BLOCK_N]
            # Compute dot(a_vec, b_block) -> [BLOCK_N]
            acc += tl.dot(a_vec, b_block)

            k_start += BLOCK_K

        # Store accumulated row to C[m, offs_n]
        c_ptrs = C_ptr + m * stride_cm + offs_n * stride_cn
        tl.store(c_ptrs, acc, mask=n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Validate inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
        assert A.shape[1] == B.shape[1], "K dimensions must match: A.shape[1] == B.shape[1]"
        M, K = A.shape
        N, K_b = B.shape
        assert K == K_b, "K mismatch"

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Allocate fp32 output for numerical stability
        C_fp32 = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)  # B is [N, K]
        stride_bk = B.stride(1)
        stride_cm = C_fp32.stride(0)
        stride_cn = C_fp32.stride(1)

        # Heuristic tile sizes
        BLOCK_N = 256 if N >= 256 else 128 if N >= 128 else 64
        BLOCK_K = 128 if K >= 128 else 64

        # Grid: one program per row
        grid = (M,)

        matmul_rowwise_kernel[grid](
            A, B, C_fp32,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3
        )

        # Cast to original dtype to match typical behavior
        if A.dtype == torch.float16:
            return C_fp32.to(torch.float16)
        elif A.dtype == torch.float32:
            return C_fp32
        else:
            # If other dtypes appear, just return fp32
            return C_fp32


def run(*args):
    return ModelNew()(*args)
