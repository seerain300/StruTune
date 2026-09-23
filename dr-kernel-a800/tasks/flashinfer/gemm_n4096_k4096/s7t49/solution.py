import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N] (PyTorch uses B.T = [N, K]). We interpret B^T[k, n] = B[n, k].
# Output C: [M, N] stored as float16 (match PyTorch default float16).
@triton.jit
def matmul_transB_3d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows k, stride_bn over cols n
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K-chunk
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k0 = pid_k * BLOCK_K
    k_offsets = k0 + tl.arange(0, BLOCK_K)           # [BLOCK_K]

    # Masks for boundary checks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = k_offsets < K  # scalar boolean broadcast

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load A_sub = A[offs_m, k_offsets] -> shape [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak)
    A_mask = mask_m[:, None] & mask_k[None, :]
    A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

    # Load B_sub as B^T[k, n] = B[n, k] -> load B[offs_n, k_offsets] -> shape [BLOCK_N, BLOCK_K]
    B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + k_offsets[None, :] * stride_bk)
    B_mask = mask_n[:, None] & mask_k[None, :]
    B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

    # Accumulate partial product for this K-chunk
    # We need acc += A_sub @ B_sub^T, where A_sub: [BLOCK_M, BLOCK_K], B_sub: [BLOCK_N, BLOCK_K]
    # Compute B_sub^T: [BLOCK_K, BLOCK_N]
    acc += tl.dot(A_sub, tl.trans(B_sub))

    # Store the result to C[offs_m, offs_n] with mask for boundaries
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = mask_m[:, None] & mask_n[None, :]
    # Store as fp16 to match PyTorch output dtype
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], output C: [M, N], float16.
    """
    assert A.is_cuda and B.is_cuda, "Triton kernel requires CUDA tensors."
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D."
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Make inputs contiguous for predictable strides
    A_c = A.contiguous()
    B_c = B.contiguous()

    # Allocate output tensor (float16 to match original get_inputs())
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Tile sizes (moderate defaults; can be tuned)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_3d_kernel[grid](
        A_c, B_c, C,
        M, N, K,
        A_c.stride(0), A_c.stride(1),
        B_c.stride(0), B_c.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
