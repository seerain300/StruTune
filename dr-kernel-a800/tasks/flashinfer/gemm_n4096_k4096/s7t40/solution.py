import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T via explicit outer-product accumulation.
# A: [M, K], B: [K, N], output C: [M, N] (float16).
@triton.jit
def matmul_transB_outer_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # strides for A (M, K)
    stride_bk, stride_bn,       # strides for B (K, N)
    stride_cm, stride_cn,       # strides for C (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D launch grid over tiles of M and N; K is handled in loop
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    # Note: we use a scalar k in range to ensure full coverage regardless of block sizes.
    for k in range(0, K):
        # Load A_vec = A[offs_m, k] -> shape [BLOCK_M]
        A_ptrs = A_ptr + (offs_m * stride_am + k * stride_ak)
        A_mask = offs_m < M
        A_vec = tl.load(A_ptrs, mask=A_mask, other=0.0)  # float16

        # Load B_vec = B[k, offs_n] -> shape [BLOCK_N] (B_T[k, n] = B[n, k], so B[k, n] is correct)
        B_ptrs = B_ptr + (k * stride_bk + offs_n * stride_bn)
        B_mask = offs_n < N
        B_vec = tl.load(B_ptrs, mask=B_mask, other=0.0)  # float16

        # Outer product accumulate in fp32
        acc += A_vec.to(tl.float32)[:, None] * B_vec.to(tl.float32)[None, :]

    # Store result to C as float16 (match PyTorch output dtype)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N]. Returns C: [M, N].
    """
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors for Triton execution."
    # Ensure contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    assert Kb == K, f"B's first dimension Kb={Kb} must match A's second dimension K={K}"

    # Allocate output C on device (float16 to match PyTorch default in get_inputs)
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides in elements
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Tile sizes: modest to keep register pressure low and correctness robust
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid over tiles
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transB_outer_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
