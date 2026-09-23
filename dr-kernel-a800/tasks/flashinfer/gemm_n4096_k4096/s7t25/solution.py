import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N] (original B), output C: [M, N] (float16)
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,              # strides for A (row-major: [M, K])
    stride_bk, stride_bn,              # strides for B (row-major: [K, N])
    stride_cm, stride_cn,              # strides for C (row-major: [M, N])
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]

    # Masks for bounds
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_mask = offs_k < K

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load A_sub tile: [BLOCK_M, BLOCK_K], A[m, k]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    A_mask = (m_mask[:, None]) & (k_mask[None, :])
    A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

    # Load B_sub_T tile: [BLOCK_N, BLOCK_K], B^T[k, n] = B[n, k]
    B_ptrs_T = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    B_mask = (n_mask[:, None]) & (k_mask[None, :])
    B_sub_T = tl.load(B_ptrs_T, mask=B_mask, other=0.0)

    # Accumulate: acc += A_sub @ B_sub_T^T  (B_sub_T^T is [BLOCK_K, BLOCK_N])
    acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub_T).to(tl.float32))

    # Store result to C in float16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (m_mask[:, None]) & (n_mask[None, :])
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N]
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: A is [{M},{K}], B is [{K2},{N}]"

    # Ensure CUDA tensors
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)

    # Make tensors contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor, float16 to match original example
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides in elements
    stride_am, stride_ak = A.stride(0), A.stride(1)   # A is [M, K]
    stride_bk, stride_bn = B.stride(0), B.stride(1)   # B is [K, N]
    stride_cm, stride_cn = C.stride(0), C.stride(1)   # C is [M, N]

    # Tiling parameters (tuned for general performance; adjust if needed)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    # 3D grid covers M, N tiles and K chunks
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Compute C = A @ B.T using Triton
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
