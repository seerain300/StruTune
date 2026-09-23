import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes C = A @ B.T
# A: [M, K], B: [N, K] (original), output C: [M, N]
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # A strides: row (M), col (K)
    stride_bn, stride_bk,     # B strides: row (N), col (K)
    stride_cm, stride_cn,     # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k

        # Load A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
        a_mask = (mask_m[:, None]) & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B as if transposed: B[k, n] using B's original strides (B is [N, K])
        # Pointer arithmetic: B[k, n] => base + n * stride_bn + k * stride_bk
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + k_idx[:, None] * stride_bk)
        b_mask = (k_idx[:, None] < K) & (mask_n[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate acc += a @ b
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # Store the result to C (fp32). Host will cast to desired dtype after kernel if needed.
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
        super().__init__()
        # Default tiling parameters; can be tuned if needed
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # If Triton or CUDA not available, fallback to torch for correctness
        if (not TRITON_AVAILABLE) or (A.device.type != 'cuda') or (B.device.type != 'cuda'):
            return torch.matmul(A, B.T)

        # Validate shapes: A is [M, K], B is [N, K]; output is [M, N]
        assert A.ndim == 2, f"A must be 2D, got shape {A.shape}"
        assert B.ndim == 2, f"B must be 2D, got shape {B.shape}"
        M, K_A = A.shape
        N_B, K_B = B.shape
        if K_A != K_B:
            # Fallback if dimensions don't align for GEMM
            return torch.matmul(A, B.T)
        N = N_B
        K = K_A

        # Ensure contiguous tensors for better performance
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (compute in fp32 for stability), then cast to input dtype
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Grid: one program per output tile
        grid = (triton.cdiv(M, self.BLOCK_M), triton.cdiv(N, self.BLOCK_N))

        # Choose number of warps based on tile size
        if self.BLOCK_M * self.BLOCK_N >= 4096:
            num_warps = 8
        else:
            num_warps = 4

        # Launch Triton kernel
        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),   # strides for A (row, col)
            B.stride(0), B.stride(1),   # strides for B (row, col)
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=num_warps, num_stages=3,
        )

        # Cast output to the input's dtype to mimic PyTorch's behavior
        if C.dtype != A.dtype:
            C = C.to(A.dtype)

        return C


def run(*args):
    return ModelNew()(*args)
