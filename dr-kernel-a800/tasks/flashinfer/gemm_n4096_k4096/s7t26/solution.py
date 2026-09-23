import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N]
# We interpret B.T as a tensor of shape [N, K] with strides (B.stride(1), B.stride(0)).
@triton.jit
def matmul_transB_kernel_3d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,          # A strides: row (M), col (K)
    stride_bt_n, stride_bt_k,      # B.T strides: row (N), col (K), where B.T is [N, K]
    stride_cm, stride_cn,          # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile offsets
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
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub_T = B.T[offs_n, offs_k] where B.T is [N, K]
        # Mapping: B.T[n, k] = B[k, n] -> B_ptr + k * B.stride(0) + n * B.stride(1)
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bt_n + offs_k[None, :] * stride_bt_k)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub_T = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_sub @ B_sub_T^T
        # B_sub_T is [BLOCK_N, BLOCK_K]; we need [BLOCK_K, BLOCK_N] for dot
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub_T.to(tl.float32)))

    # Store result to C as float16 (original tensors are float16)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, {N}]"
    # Ensure CUDA tensors
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)
    # Ensure contiguous for stride simplicity
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor, float16 to match original get_inputs dtype
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides in element units
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    # B.T is [N, K]; strides: row=N, col=K
    stride_bt_n = B.stride(1)  # stride along N for B.T
    stride_bt_k = B.stride(0)  # stride along K for B.T
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Tiling parameters (safe defaults; correct across a wide range of shapes)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    # 3D grid covers all M, N, K
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel_3d[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bt_n, stride_bt_k,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Use Triton kernel to compute A @ B.T
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
