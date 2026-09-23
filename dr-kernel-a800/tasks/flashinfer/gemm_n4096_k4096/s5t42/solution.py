import torch
import triton
import triton.language as tl

# 2D GEMM kernel: C[M, N] = A[M, K] @ BT[K, N], where BT = B.T contiguous [N, K]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bt_kernel(
    A_ptr,           # *fp16, shape [M, K], contiguous
    BT_ptr,          # *fp16, shape [N, K], contiguous (B.T)
    C_ptr,           # *fp16, shape [M, N], contiguous
    M, N, K,         # runtime sizes (not constexpr to allow autotune key)
    stride_am, stride_ak,   # strides for A: (row, col)
    stride_bn, stride_bk,   # strides for BT: (row n, col k)
    stride_cm, stride_cn,   # strides for C: (row m, col n)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row/col offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k] -> [BM, BK]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: BT[n, k] = B.T[n, k], where BT is [N, K] contiguous
        # This accesses B[k, n] logically but via transposed layout
        BT_ptrs = BT_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        bt_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate: a [BM, BK] dot bt [BK, BN] -> [BM, BN]
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Store results to C[m, n] (fp16) with proper masking
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and float16
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Enforce contiguity for robust stride handling
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Shapes: A [M, K], B [K, N]
        M, K = A_c.shape
        Kb, N = B_c.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). B's K must equal A's K.")

        # Materialize B.T as contiguous [N, K] to avoid stride complexities in the kernel
        BT = B_c.t().contiguous()  # [N, K], fp16

        # Allocate output tensor C [M, N], fp16, contiguous
        C = torch.empty((M, N), dtype=torch.float16, device=A_c.device).contiguous()

        # 2D grid over tiles of M and N; masks handle any out-of-bounds rows/cols
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
