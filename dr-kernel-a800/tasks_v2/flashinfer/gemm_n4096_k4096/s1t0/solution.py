import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes C = A @ B.T where
# A: [M, K], B: [N, K] (original), B.T is conceptually [K, N], output: [M, N]
# We avoid materializing B.T by using B's strides to read as if transposed.
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: row (M), col (K)
    stride_bn, stride_bk,   # B strides: row (N), col (K)
    stride_cm, stride_cn,   # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create masks for bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Compute k indices for this block
        k_idx = k + offs_k

        # Pointers for A: A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
        a_mask = (mask_m[:, None]) & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B as if transposed: B[k, n] using original B strides (B is [N, K])
        # Reading B[k, n] means: row in B is k (second dim of B), column is n (first dim).
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + k_idx[:, None] * stride_bk)
        b_mask = (k_idx[:, None] < K) & (mask_n[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: acc += a @ b
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # Store results to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    # We accumulate in fp32, and we can cast to fp16 for output if desired.
    # Here we store fp32 and rely on the host to cast to desired dtype if needed.
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes: good defaults for fp16 GEMM
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 32

    def forward(self, A, B):
        # Ensure Triton is available and tensors are on CUDA
        if (not TRITON_AVAILABLE) or (not A.is_cuda) or (not B.is_cuda):
            # Fallback to PyTorch if Triton/CUDA is not available
            return torch.matmul(A, B.T)

        # Validate shapes: A is [M, K], B is [N, K] originally, we need B.T [K, N], output [M, N]
        if A.dim() != 2 or B.dim() != 2:
            # General fallback if shapes are unexpected
            return torch.matmul(A, B.T)

        M, K_A = A.shape
        N_B, K_B = B.shape

        # The original code uses B.T, so the second dimension of B must match A's second dim for matmul to be valid.
        # In other words, K_A must equal N_B. If not, PyTorch would error; we mimic that behavior.
        if K_A != N_B:
            # Fallback to PyTorch for correctness (shape mismatch)
            return torch.matmul(A, B.T)

        # We can allow K_A != K_B as long as we conceptualize B.T as [K, N], but B's original shape [N, K] means K_B should be K.
        # The original code doesn't enforce B's first dim to match A's second dim; it just uses B.T.
        # To be safe, we'll require B's second dim (K) to be the same as A's second dim (K_A). If not, fallback.
        # However, in typical matmul(A, B.T), B is [N, K], and B.T is [K, N], so K_B is the reduction dim (K) and N_B is output columns (N).
        # The operation is valid as long as K_A == K_B (i.e., A's reduction dim matches B's original second dim).
        if K_A != K_B:
            # Fallback if dimensions don't align for GEMM
            return torch.matmul(A, B.T)

        # Ensure contiguous for performance
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor: [M, N_B] (N = N_B)
        # We'll compute in fp32 and cast to desired dtype after (to mimic input dtype behavior).
        out_dtype = A.dtype  # default to A's dtype; original code returns same dtype
        M, K = A.shape
        N = N_B  # number of columns in B (output N)

        # Allocate output as fp32 accumulator
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Compute grid
        grid = (triton.cdiv(M, self.BLOCK_M), triton.cdiv(N, self.BLOCK_N))

        # Choose number of warps: simple heuristic
        # Larger tiles -> more warps
        if self.BLOCK_M * self.BLOCK_N >= 4096:
            num_warps = 8
        else:
            num_warps = 4

        # Launch kernel
        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),   # A strides: row (M), col (K)
            B.stride(0), B.stride(1),   # B strides: row (N_B), col (K_B)
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=num_warps, num_stages=3,
        )

        # Cast output to original dtype if needed
        if out_dtype != torch.float32:
            C = C.to(out_dtype)

        # Return tensor (the original run returns a tensor; we return tensor)
        return C


def run(*args):
    return ModelNew()(*args)
