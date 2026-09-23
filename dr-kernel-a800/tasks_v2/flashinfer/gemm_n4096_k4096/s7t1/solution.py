import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [N, K] (note: original PyTorch uses B.T, so B is [N, K] in matmul(A, B.T)).
# Output C: [M, N]
@triton.jit
def matmul_transB_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # B is [N, K]: stride_bn over rows (n), stride_bk over cols (k)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offsets within the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for better numerical behavior
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_sub as B^T: B^T[n, k] = B[n, k], but we use B as [N, K] and compute B.T implicitly
        # B[n, k] -> we load B[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate dot product: (BLOCK_M, BLOCK_K) @ (BLOCK_N, BLOCK_K)^T -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(A_sub.to(tl.float32), tl.trans(B_sub).to(tl.float32))

    # Store results to C at [offs_m, offs_n], cast acc (fp32) to the dtype of C_ptr (float16 expected)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes: robust defaults
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 32

    def forward(self, A, B):
        # Move to CUDA (Triton requires GPU tensors)
        device = torch.device('cuda')
        A_dev = A.to(device, non_blocking=True)
        B_dev = B.to(device, non_blocking=True)

        # Ensure contiguous for coalesced access
        A_dev = A_dev.contiguous()
        B_dev = B_dev.contiguous()

        M, K = A_dev.shape
        # B should be [N, K] for torch.matmul(A, B.T)
        N = B_dev.shape[0]  # number of rows of B, which corresponds to N in B.T

        # Output tensor: [M, N], float16 to match original behavior
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
            B_dev.stride(0), B_dev.stride(1),  # B is [N, K]
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Return result (on CUDA)
        return C


def run(*args):
    return ModelNew()(*args)
