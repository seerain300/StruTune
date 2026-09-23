import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N] (PyTorch uses B.T = [N, K]). We interpret B^T[k, n] = B[n, k].
# Output C: [M, N], float16.
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows k, stride_bn over cols n
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K-chunks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile offsets in M and N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K, ensuring full coverage
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Perform matrix multiply for this tile: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub).to(tl.float32))

        k0 += BLOCK_K

    # Store result to C. Output is float16, so cast acc to fp16.
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Host-side wrapper: ensures CUDA/contiguous and launches the Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA for Triton kernel."
    # Make sure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]"
    # Allocate output C on same device
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides in element units
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tuned tiling: balance occupancy and register pressure
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64  # reduce K-chunk iterations (K=4096 -> 64 chunks)

    # Grid must cover all dimensions; third dim covers K in chunks
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4,
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
