import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B^T
# A: [M, K], B: [K, N] (PyTorch uses B.T = [N, K]). We interpret B^T[k, n] = B[n, k].
# Output C: [M, N], float16 (PyTorch original uses float16).
@triton.jit
def matmul_transB_kernel_3d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]: stride_bk over rows k, stride_bn over cols n
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Output tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute K chunk offsets for this program
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)

    # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

    # Load B_sub as B^T: B^T[k, n] = B[n, k] -> load B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
    B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

    # Accumulate dot product: (BLOCK_M, BLOCK_K) @ (BLOCK_N, BLOCK_K)^T -> (BLOCK_M, BLOCK_N)
    acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub).to(tl.float32))

    # Store results to C at [offs_m, offs_n], cast acc (fp32) to fp16 for output
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)  # Triton will cast fp32 to fp16 on store


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tuned parameters for performance on large N and small M
        self.BLOCK_M = 64
        self.BLOCK_N = 128
        self.BLOCK_K = 128
        self.num_warps = 8
        self.num_stages = 4

    def forward(self, A, B):
        # Ensure inputs are on CUDA device (Triton requires GPU tensors)
        device = torch.device('cuda')
        A_dev = A.to(device, non_blocking=True)
        B_dev = B.to(device, non_blocking=True)

        # Ensure contiguous for coalesced access
        A_dev = A_dev.contiguous()
        B_dev = B_dev.contiguous()

        M, K = A_dev.shape
        # B must be [K, N] for A @ B^T
        N = B_dev.shape[1]

        # Output tensor: [M, N], float16 to match original
        C = torch.empty((M, N), device=device, dtype=torch.float16)

        # 3D launch grid over M, N, and K tiles
        grid = (
            triton.cdiv(M, self.BLOCK_M),
            triton.cdiv(N, self.BLOCK_N),
            triton.cdiv(K, self.BLOCK_K),
        )

        # Launch Triton kernel
        matmul_transB_kernel_3d[grid](
            A_dev, B_dev, C,
            M, N, K,
            A_dev.stride(0), A_dev.stride(1),
            B_dev.stride(0), B_dev.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # Return result (on CUDA)
        return C


def run(*args):
    return ModelNew()(*args)
