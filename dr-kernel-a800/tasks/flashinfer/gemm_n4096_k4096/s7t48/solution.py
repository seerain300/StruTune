import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T, where A: [M, K], B: [K, N], output C: [M, N]
@triton.jit
def matmul_transB_kernel_3d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K chunk start for this program
    k0 = pid_k * BLOCK_K
    k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # Masks for boundary checks
    mask_m = offs_m < M                        # [BLOCK_M]
    mask_n = offs_n < N                        # [BLOCK_N]
    mask_k_scalar = k_offsets < K              # [BLOCK_K], used for loading/storing in K-chunk

    # Loop over K chunk (vectorized). We construct pointers for each k in the chunk.
    # Note: We build A_sub [BLOCK_M, BLOCK_K] and B_sub [BLOCK_K, BLOCK_N] and accumulate.
    for kk in range(0, BLOCK_K):
        k = k_offsets[kk]  # scalar k in this chunk

        # Compute pointers and masks for this k
        # A_sub = A[offs_m, k] -> shape [BLOCK_M, 1]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + k * stride_ak)  # [BLOCK_M, 1]
        A_mask = mask_m[:, None] & mask_k_scalar[kk]                     # [BLOCK_M, 1]
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)   # [BLOCK_M, 1]

        # B_sub = B_T[k, offs_n] = B[offs_n, k] -> shape [1, BLOCK_N]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + k * stride_bk)   # [1, BLOCK_N]
        B_mask = mask_n[None, :] & mask_k_scalar[kk]                     # [1, BLOCK_N]
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)   # [1, BLOCK_N]

        # Accumulate: acc += A_sub @ B_sub
        # A_sub: [BLOCK_M, 1], B_sub: [1, BLOCK_N] -> result [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_sub, tl.trans(B_sub))

    # Store result back to C in float16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)  # [BLOCK_M, BLOCK_N]
    C_mask = mask_m[:, None] & mask_n[None, :]                                      # [BLOCK_M, BLOCK_N]
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are on CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Choose tile sizes. These are reasonable defaults for fp16 matmul.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # 3D grid over (M, N, K) tiles
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel_3d[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Move tensors to CUDA if needed (Triton requires CUDA)
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
