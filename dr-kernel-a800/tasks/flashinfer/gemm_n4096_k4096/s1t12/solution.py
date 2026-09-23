import torch
import triton
import triton.language as tl


# Robust 2D tiled Triton kernel computing C = A @ B.T
# A: [M, K], B: [N, K], C: [M, N]
@triton.jit
def matmul_bt_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)        # [BLOCK_M]
    offs_n = n_start + tl.arange(0, BLOCK_N)        # [BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)    # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load B block as B[k, n] to emulate B.T without materializing: indices are (n, k)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)  # [BLOCK_K, BLOCK_N]
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += a_tile @ b_block
        acc += tl.dot(a_tile, b_block)

    # Store results C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        # Ensure contiguity
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's second dim ({Kb}) must match A's K ({K})."

        # Output in fp32 for numerical robustness; cast to A.dtype at the end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)
        stride_bk = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid over M and N tiles
        # Tuned for good throughput on large matrices; adjust if needed.
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_bt_tiled_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3
        )

        # Cast result to input A dtype to match original behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
