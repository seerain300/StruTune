import torch
import triton
import triton.language as tl

# Triton kernel: C[M, N] = A[M, K] @ BT[K, N], where BT = B.T contiguous [N, K]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bT_kernel(
    A_ptr,        # *fp16, A[M, K]
    BT_ptr,       # *fp16, BT[N, K] = B.T contiguous
    C_ptr,        # *fp16, output C[M, N]
    M, N, K,      # int32 sizes
    stride_am, stride_ak,   # A strides
    stride_bn, stride_bk,   # BT strides (BN is stride along N, BK along K)
    stride_cm, stride_cn,   # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute program IDs with dynamic coverage
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m = tl.num_programs(0)
    total_n = tl.num_programs(1)

    # Tile offsets
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

        # Pointers for BT tile: BT[n, k] (note: BT is [N, K])
        BT_ptrs = BT_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        bt_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results to C[m, n], cast to fp16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Shapes: A [M, K], B [K, N]
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}). B's K must equal A's K.")

        # Enforce dtype and contiguity for robust indexing
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        A_c = A.contiguous()
        B_c = B.contiguous()

        # Materialize BT = B.T as a contiguous [N, K] tensor (fp16), for simple, correct indexing
        BT = B_c.t().contiguous()  # [N, K]

        # Output tensor, contiguous, float16. Shape (M, N).
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Launch Triton kernel with a grid that fully covers M and N tiles
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        _matmul_bT_kernel[grid](
            A_c, BT, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
        )
        return C


def run(*args):
    return ModelNew()(*args)
