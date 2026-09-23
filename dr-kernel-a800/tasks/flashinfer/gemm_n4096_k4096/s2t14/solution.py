import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M/N
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8,  num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bt_kernel(
    A_ptr,  # *fp16, shape [M, K]
    B_ptr,  # *fp16, shape [K, N]
    C_ptr,  # *fp16, shape [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k in tl.static_range(0, K, BLOCK_K):
        # Compute current K indices for this chunk
        k_idx = k + offs_k  # shape [BLOCK_K]
        # Load A tile: A[offs_m, k_idx] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K) + k_idx[None, :]
        a_mask = mask_m[:, None] & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype follows A_ptr, typically fp16

        # Load B tile: B[k_idx, offs_n] -> emulate B.T by using B's strides:
        # B[k, n] => B_ptr + k*B.stride(0) + n*B.stride(1)
        b_ptrs = B_ptr + (k_idx[:, None] * N) + offs_n[None, :]
        b_mask = (k_idx[:, None] < K) & mask_n[None, :]
        # Note: For B of shape [K, N], B[k, n] element accessed as above; this corresponds to
        # B.T[n, k] with strides (B.stride(1), B.stride(0)). We don't materialize B.T.
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store results to C (fp16), with boundary masks
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are contiguous and on CUDA
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device for Triton kernel."
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Output tensor (fp16, matching input dtype)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Define grid based on tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel (autotuner will choose best config)
        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
        )
        return C


def run(*args):
    return ModelNew()(*args)
