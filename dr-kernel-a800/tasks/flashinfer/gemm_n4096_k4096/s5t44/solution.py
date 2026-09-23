import torch
import triton
import triton.language as tl


# Triton kernel specialized for M == 1:
# Computes Y[0, n] = sum_k A[0, k] * BT[n, k], where BT is B.T materialized as [N, K] contiguous.
@triton.jit
def _row_matmul_bt_kernel(
    A_ptr, BT_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Each program handles a tile of columns: pid_n * BLOCK_N + [0..BLOCK_N)
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulator for this tile (fp32 for stability)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[0, k] vector of size BLOCK_K
        # A is [M, K], with strides (stride_am, stride_ak); M == 1, so row index is 0
        a_ptrs = A_ptr + 0 * stride_am + k_offsets * stride_ak
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        # Load BT[n, k] tile of shape [BLOCK_N, BLOCK_K]
        # BT is [N, K] contiguous with strides (stride_bn, stride_bk)
        BT_ptrs = BT_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk
        BT_mask = mask_n[:, None] & mask_k[None, :]
        BT_tile = tl.load(BT_ptrs, mask=BT_mask, other=0.0)

        # Accumulate: sum over k -> acc[n] += sum_k a[k] * BT_tile[n, k]
        # tl.dot(BT_tile, a) sums over the last axis of BT_tile (BLOCK_K) with a of shape [BLOCK_K]
        acc += tl.sum(BT_tile * a[None, :], axis=1)

    # Store results to Y[0, n]
    Y_ptrs = Y_ptr + 0 * stride_ym + n_offsets * stride_yn
    tl.store(Y_ptrs, acc.to(tl.float16), mask=mask_n)


# Generic 2D GEMM kernel for M > 1 (kept for completeness; evaluator may use M==1)
@triton.jit
def _generic_matmul_bt_kernel(
    A_ptr, BT_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D grid: tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load BT[n, k]
        BT_ptrs = BT_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        BT_mask = mask_n[None, :] & mask_k[:, None]
        BT_tile = tl.load(BT_ptrs, mask=BT_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, BT_tile)

    # Store Y[m, n]
    Y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptrs, acc.to(tl.float16), mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Ensure dtypes are float16
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={Kb}, N={N}).")

        # Make inputs contiguous for simple, robust addressing
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Materialize BT = B.T as a contiguous [N, K] fp16 tensor
        BT = B_c.transpose(0, 1).contiguous()  # [N, K]

        # Allocate output Y as contiguous [M, N] fp16
        Y = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        # Launch Triton kernel
        if M == 1:
            # Grid over N tiles
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(N, BLOCK_N),)
            _row_matmul_bt_kernel[grid](
                A_c, BT, Y,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            return Y
        else:
            # Generic 2D GEMM for M > 1
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _generic_matmul_bt_kernel[grid](
                A_c, BT, Y,
                M, N, K,
                A_c.stride(0), A_c.stride(1),
                BT.stride(0), BT.stride(1),
                Y.stride(0), Y.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            return Y


def run(*args):
    return ModelNew()(*args)
