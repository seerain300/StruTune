import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N]
# In this kernel, we interpret B^T[k, n] = B[n, k].
@triton.jit
def matmul_transB_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows (k), stride_bn over cols (n)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of M and N
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

        # Load A_sub = A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: A_sub is [BM, BK], tl.trans(B_sub) is [BK, BN]
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub.to(tl.float32)))

    # Store results with masks for boundaries
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store as float16 (C is fp16). acc is fp32; Triton will cast.
    tl.store(C_ptrs, acc, mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N], return C: [M, N]
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [{M}, {K}] and B is [{Kb}, {N}]"
    # Ensure CUDA tensors and contiguity
    A = A.contiguous()
    B = B.contiguous()
    # Output tensor in float16 to match get_inputs default
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides in elements
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)  # along K (rows of B)
    stride_bn = B.stride(1)  # along N (cols of B)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Tuned tiling (can be adjusted based on hardware)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transB_kernel_2d[grid](
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
        # Triton requires CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
