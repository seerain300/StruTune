import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T, where
# A is [M, K], B is [K, N], output C is [M, N]
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: tiles across M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Output tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub = B[offs_k, offs_n] -> shape [BLOCK_K, BLOCK_N], corresponds to B^T[n, k] = B[k, n]
        B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate dot product in fp32
        acc += tl.dot(A_sub.to(tl.float32), B_sub.to(tl.float32))

    # Store results to C at [offs_m, offs_n]
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)  # acc is fp32, Triton will cast to C's dtype on store


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes; 64x64x32 is a robust default
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 32

    def forward(self, A, B):
        # Ensure inputs are on CUDA (Triton requires GPU)
        device = torch.device('cuda')
        A_dev = A.to(device, non_blocking=True)
        B_dev = B.to(device, non_blocking=True)
        # Ensure contiguous for coalesced access
        A_dev = A_dev.contiguous()
        B_dev = B_dev.contiguous()

        M, K = A_dev.shape
        Kb, N = B_dev.shape

        # Output tensor: [M, N], float16 to match original
        C = torch.empty((M, N), device=device, dtype=torch.float16)

        # Launch grid
        grid = (
            triton.cdiv(M, self.BLOCK_M),
            triton.cdiv(N, self.BLOCK_N),
            triton.cdiv(K, self.BLOCK_K),
        )

        # Launch Triton kernel
        matmul_transB_kernel[grid](
            A_dev, B_dev, C,
            M, N, K,
            A_dev.stride(0), A_dev.stride(1),
            B_dev.stride(0), B_dev.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Return result on CUDA
        return C


def run(*args):
    return ModelNew()(*args)
