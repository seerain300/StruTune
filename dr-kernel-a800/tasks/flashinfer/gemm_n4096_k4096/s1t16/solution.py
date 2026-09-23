import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Row-wise kernel: compute one output row vector c[m, :] = A[m, :] @ B.T
# Grid: (M,)
@triton.jit
def _matmul_rowwise_fp32(A_ptr, B_ptr, C_ptr,
                          M, N, K,
                          stride_am, stride_ak,
                          stride_bn, stride_bk,
                          stride_cm, stride_cn,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    # Vector of column indices
    cols = tl.arange(0, BLOCK_N)
    # Accumulator for the row
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over N in tiles
    n_start = 0
    while n_start < N:
        n_offsets = n_start + cols
        # Loop over K in tiles
        k_start = 0
        while k_start < K:
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Load A[m, k_offsets]
            a_ptrs = A_ptr + m * stride_am + k_offsets * stride_ak
            a = tl.load(a_ptrs, mask=k_offsets < K, other=0.0)
            # Load B[n_offsets, k_offsets] to form partial [BLOCK_N x BLOCK_K]
            b_ptrs = B_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk
            b_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            # acc += a[:, None] @ b[None, :]
            acc += tl.sum(b * a[:, None], axis=0)
            k_start += BLOCK_K
        n_start += BLOCK_N

    # Store result row to C[m, :]
    c_ptrs = C_ptr + m * stride_cm + cols * stride_cn
    store_mask = (m < M) & (cols < N)
    tl.store(c_ptrs, acc, mask=store_mask)


# General 2D-tiled kernel: compute C[m, n] tile for m in [pid_m*BM : (pid_m+1)*BM), n in [pid_n*BN : (pid_n+1)*BN)
# Grid: (ceil_div(M, BM), ceil_div(N, BN))
@triton.jit
def _matmul_bt_2d_fp32(A_ptr, B_ptr, C_ptr,
                       M, N, K,
                       stride_am, stride_ak,
                       stride_bn, stride_bk,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile: we index B[n, k] to get [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)
        k_start += BLOCK_K

    # Store C[m, n] tile
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure Triton is available and tensors are on CUDA
        if (not TRITON_AVAILABLE) or (not A.is_cuda) or (not B.is_cuda):
            # Fallback to PyTorch for CPU or missing Triton
            return torch.matmul(A, B.T)

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K_a = A.shape
        N_b, K_b = B.shape
        assert K_a == K_b, "Inner dimension K must match for A and B"

        K = K_a  # equals K_b
        N = N_b

        # Output as fp32 for stable accumulation
        C_fp32 = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides for output
        stride_cm = C_fp32.stride(0)
        stride_cn = C_fp32.stride(1)

        # Strides for A and B
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)
        stride_bk = B.stride(1)

        # Tuned constants
        # Row-wise kernel parameters
        BLOCK_N_ROW = 256
        BLOCK_K_ROW = 128
        NUM_WARPS_ROW = 4
        NUM_STAGES_ROW = 2

        # 2D-tiled kernel parameters
        BLOCK_M_2D = 128
        BLOCK_N_2D = 128
        BLOCK_K_2D = 64
        NUM_WARPS_2D = 8
        NUM_STAGES_2D = 3

        if M <= 32:
            # Row-wise kernel: one program per row
            grid = (M,)
            _matmul_rowwise_fp32[grid](
                A, B, C_fp32,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N_ROW, BLOCK_K=BLOCK_K_ROW,
                num_warps=NUM_WARPS_ROW, num_stages=NUM_STAGES_ROW,
            )
        else:
            # 2D-tiled kernel
            grid = (triton.cdiv(M, BLOCK_M_2D), triton.cdiv(N, BLOCK_N_2D))
            _matmul_bt_2d_fp32[grid](
                A, B, C_fp32,
                M, N, K,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M_2D, BLOCK_N=BLOCK_N_2D, BLOCK_K=BLOCK_K_2D,
                num_warps=NUM_WARPS_2D, num_stages=NUM_STAGES_2D,
            )

        # Return fp32 output
        return C_fp32


def run(*args):
    return ModelNew()(*args)
