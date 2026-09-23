import torch
import triton
import triton.language as tl

# Robust 2D-tiled matmul kernel computing C = A @ B.T
# A: [M, K], B: [N, K], C: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_at_btrans_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # Mask for A loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile corresponding to B_T[k, n] = B[n, k]:
        # Treat B as [N, K], so indexing B[n, k] with n=offs_n (columns of BT) and k=offs_k (rows of BT).
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        # Mask for B loads
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store as fp32 then cast to original dtype (assume fp16 in typical usage)
    # Triton will cast on store if C_ptr is fp16; here we keep fp32 accumulation and let Triton handle cast.
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"Incompatible shapes: A is [M, K], B is [N, {N}] but second dim must be [N, K] where K={K}"

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k]:
        stride_bn = B.stride(0)  # original B's first dim (N)
        stride_bk = B.stride(1)  # original B's second dim (K)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Define grid as a function of meta (BLOCK sizes)
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel
        matmul_at_btrans_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
