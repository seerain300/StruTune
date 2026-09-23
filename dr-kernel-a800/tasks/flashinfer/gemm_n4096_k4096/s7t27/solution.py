import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], B.T: [N, K], indexing B.T via B's strides: (B.stride(1), B.stride(0))
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # for B_T[n, k] use (B.stride(1), B.stride(0))
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]

    # Masks for boundaries
    A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load A_sub = A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

    # Load B_sub_T = B_T[offs_n, offs_k] = B[offs_k, offs_n] -> [BLOCK_N, BLOCK_K]
    B_ptrs_T = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    B_sub_T = tl.load(B_ptrs_T, mask=B_mask, other=0.0)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Multiply: acc += A_sub @ B_sub_T^T
    # B_sub_T is [BLOCK_N, BLOCK_K]; transpose to [BLOCK_K, BLOCK_N] for dot with A_sub [BLOCK_M, BLOCK_K]
    acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub_T).to(tl.float32))

    # Store to C[offs_m, offs_n] as float16 (to match get_inputs dtype)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], output C: [M, N], float16.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA for Triton execution."
    # Ensure contiguous tensors for straightforward stride usage
    A = A.contiguous()
    B = B.contiguous()

    M, K_A = A.shape
    K_B, N = B.shape
    assert K_A == K_B, f"Incompatible shapes: A is (*, {K_A}), B is (*, {K_B})."
    K = K_A

    # Allocate output as float16
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Conservative tile sizes for safety across diverse shapes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),  # indexing B_T[n, k] via B's strides
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
