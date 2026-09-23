import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N], float16 (to match original get_inputs)
@triton.jit
def matmul_transB_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride over k (rows) and n (cols)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over tiles of M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        # Ensure mask is boolean (tl.int1) to avoid Triton compilation issues
        A_mask = A_mask.to(tl.int1)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)  # [BLOCK_K, BLOCK_N]
        B_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_mask = B_mask.to(tl.int1)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: A_sub [BLOCK_M, BLOCK_K] x B_sub^T [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_sub, tl.trans(B_sub))

    # Store result to C with boundary masks
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_mask = C_mask.to(tl.int1)
    # Cast to float16 to match output dtype (original get_inputs uses float16)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N]
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"
    # Ensure tensors are on CUDA for Triton
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    # Make inputs contiguous
    A = A.contiguous()
    B = B.contiguous()
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Tiling parameters: tuned for general performance
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transB_2d_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure we use Triton for all computation; no torch.matmul here
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
