import torch
import triton
import triton.language as tl

# Autotuned Triton kernel for C = A @ BT, where:
# A: [M, K], BT: [K, N] (BT = B.t().contiguous())
# Output C: [M, N], float16
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 1024,'BLOCK_K': 256}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Pointers for BT tile: BT[k, n] (note: BT is [K, N])
        BT_ptrs = BT_ptr + k_offsets[:, None] * stride_btk + n_offsets[None, :] * stride_btn
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results to C[m, n] as fp16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Validate shapes: A [M, K], B [K, N]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Inputs must be 2D: A [M, K], B [K, N].")
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Enforce contiguity for robust stride handling
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Materialize BT = B.t() as [K, N], float16, contiguous
        BT = B_c.t().contiguous().to(torch.float16)

        # Output tensor: [M, N], float16, contiguous
        C = torch.empty((M, N), dtype=torch.float16, device=A_c.device)

        # Define grid based on meta (BLOCK_M, BLOCK_N) for autotune
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel
        _gemm_kernel[grid](
            A_c, BT, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
        )
        return C


def run(*args):
    return ModelNew()(*args)
