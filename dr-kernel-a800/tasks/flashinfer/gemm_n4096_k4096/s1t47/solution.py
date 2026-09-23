import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Favor larger tiles along N for big outputs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64 }, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64 }, num_warps=8, num_stages=4),
        # Balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64 }, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64 }, num_warps=4, num_stages=3),
    ],
    key=['M', 'K', 'N']
)
@triton.jit
def matmul_bt_kernel(A_ptr, B_ptr, C_ptr,
                     M, N, K,
                     stride_am, stride_ak,
                     stride_bn, stride_bk,
                     stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B.T where:
      A: [M, K], B: [N, K], B.T: [K, N], C: [M, N]
    We index B as B[n, k] directly (no materialization of B.T).
    Accumulate in fp32 and store fp32 output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile corresponding to B.T[k, n] = B[n, k]:
        # We need [BLOCK_K, BLOCK_N] tile, so load B[n, k] with n along N, k along K.
        bt_ptrs = B_ptr + offs_n[None, :] * stride_bn + k_ids[:, None] * stride_bk
        bt_mask = (offs_n[None, :] < N) & (k_ids[:, None] < K)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        """
        A: [M, K], float16
        B: [N, K], float16
        Returns C: [M, N], float32
        """
        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        M, K = A.shape
        N, K_b = B.shape
        assert K == K_b, "Incompatible shapes for A and B"

        # Ensure contiguity
        A = A.contiguous()
        B = B.contiguous()

        # Output in fp32 for numerical stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)  # along N in B
        stride_bk = B.stride(1)  # along K in B
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid based on chosen BLOCK sizes; autotune will pick best config
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        matmul_bt_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn)

        return C


def run(*args):
    return ModelNew()(*args)
