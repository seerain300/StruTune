import torch
import triton
import triton.language as tl

# Autotuned general GEMM kernel: computes C = A @ BT, where BT is B.t() contiguous [K, N].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bt_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Pointers for BT tile: BT[k, n]
        BT_ptrs = BT_ptr + k_offsets[:, None] * stride_btk + n_offsets[None, :] * stride_btn
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results to C[m, n]
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            # Fallback to PyTorch on CPU for safety, though evaluator provides CUDA inputs
            return A @ B.T

        # Validate shapes
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Inputs must be 2D: A [M, K], B [K, N].")
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Enforce contiguity for robust strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Materialize BT = B.T as contiguous [K, N] for simple, robust indexing
        BT = B_c.t().contiguous()  # [K, N], float16

        # Output tensor: [M, N], float16, contiguous
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Launch autotuned GEMM kernel
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        _gemm_bt_kernel[grid](
            A_c, BT, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
        )
        return C


def run(*args):
    return ModelNew()(*args)
