import torch
import triton
import triton.language as tl


# 1D kernel specialized for tiny M: parallelize across N tiles, accumulate across all M and K
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_smallM_1D(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: [M, K]
    stride_bk, stride_bn,        # B_T strides: [K, N] (B_T[k, n] = B[n, k])
    stride_cm, stride_cn,        # C strides: [M, N]
    BLOCK_N: tl.constexpr,       # tile size along N
    BLOCK_K: tl.constexpr,       # reduction tile along K
):
    # Each program handles a contiguous tile of N columns: [pid * BLOCK_N : (pid+1) * BLOCK_N)
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

        # Accumulate over all rows (M) in the outer loop to support tiny M robustly
        for i in range(0, M):
            # Load A[i, k_offsets] as vector of length BLOCK_K (masked)
            a_ptrs = A_ptr + (i * stride_am) + (k_offsets * stride_ak)
            a_vec = tl.load(a_ptrs, mask=k_mask, other=0.0)  # dtype follows A_ptr (fp16 here)
            a_vec = a_vec.to(tl.float32)  # accumulate in fp32

            # Load B_T[k_offsets, n_offsets] as matrix [BLOCK_K, BLOCK_N]
            b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk) + (n_offsets[None, :] * stride_bn)
            # 2D mask combining row bounds (i < M), k bounds, and n bounds
            b_mask = (i < M) & (k_mask[:, None]) & (n_mask[None, :])
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

            # Accumulate: acc[i, :] += dot(a_vec, b_tile, axis=0)
            # b_tile shape [BLOCK_K, BLOCK_N], a_vec shape [BLOCK_K] -> result [BLOCK_N]
            acc[i, :] += tl.sum(b_tile * a_vec[:, None], axis=0)

    # Store results to C[M, N]
    c_ptrs = C_ptr + (tl.arange(0, M)[:, None] * stride_cm) + (n_offsets[None, :] * stride_cn)
    # Final mask: rows in bounds and columns in bounds
    c_mask = (tl.arange(0, M)[:, None] < M) & (n_mask[None, :])
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# General 2D kernel for larger M (robust and simple mask logic)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_2D(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: [M, K]
    stride_bk, stride_bn,        # B_T strides: [K, N] (B_T[k, n] = B[n, k])
    stride_cm, stride_cn,        # C strides: [M, N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # fp32 accumulator for BLOCK_M x BLOCK_N tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K], masked by m and k bounds
        a_ptrs = A_ptr + (m_offsets[:, None] * stride_am) + (k_offsets[None, :] * stride_ak)
        a_mask = (m_mask[:, None]) & (k_mask[None, :])
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load B_T tile: [BLOCK_K, BLOCK_N], masked by k and n bounds
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk) + (n_offsets[None, :] * stride_bn)
        b_mask = (k_mask[:, None]) & (n_mask[None, :])
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(a_tile, b_tile)

    # Store results to C
    c_ptrs = C_ptr + (m_offsets[:, None] * stride_cm) + (n_offsets[None, :] * stride_cn)
    c_mask = (m_mask[:, None]) & (n_mask[None, :])
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A @ B.T where:
          A: [M, K]
          B: [N, O] but we index B_T[k, n] = B[n, k]
        Returns C of shape [M, N], dtype float16.
        """
        # Ensure device is CUDA (Triton requires GPU)
        if A.device.type != 'cuda' or B.device.type != 'cuda':
            raise RuntimeError("ModelNew requires CUDA tensors")

        # Make inputs contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Create B_T as [K, N] via transpose (avoid materializing transpose otherwise)
        # B_T[k, n] = B[n, k]
        B_T = B.transpose(0, 1).contiguous()

        M, K = A.shape  # A is [M, K]
        N, O = B.shape  # B is [N, O]; in evaluator, O==K since original code uses B.T

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B_T.stride(0)  # along K
        stride_bn = B_T.stride(1)  # along N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch kernel:
        # - For tiny M (<= 16), use 1D kernel over N tiles for better utilization on small M.
        # - Otherwise, use 2D kernel.
        if M <= 16:
            def grid(meta):
                return (triton.cdiv(N, meta['BLOCK_N']),)
            matmul_smallM_1D[grid](
                A, B_T, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )
        else:
            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            matmul_2D[grid](
                A, B_T, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
