import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes C = A @ B.T
# A: [M, K_a], B: [N_b, K_b], output C: [M, N_b], stored as fp32 for numerical stability
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: row (M), col (K)
    stride_bn, stride_bk,   # B strides: row (N), col (K)
    stride_cm, stride_cn,   # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along N
    BLOCK_K: tl.constexpr,  # tile size along K
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype follows A_ptr (usually fp16/bf16/fp32)

        # Pointers for B tile treated as B_T [BLOCK_K, BLOCK_N] by indexing B[k, n] -> B_ptr + n*stride_bn + k*stride_bk
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype follows B_ptr

        # Accumulate
        # a is [BLOCK_M, BLOCK_K], b is [BLOCK_K, BLOCK_N] -> acc += a @ b
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store results to C (fp32)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs are 2D
        if not (A.ndim == 2 and B.ndim == 2):
            raise ValueError("A and B must be 2D tensors")
        M, K_a = A.shape
        N_b, K_b = B.shape
        if K_a != K_b:
            raise ValueError(f"Cannot multiply A[M, K_a]={A.shape} with B[N_b, K_b]={B.shape}: K dimensions must match")

        # Triton and CUDA must be available; otherwise raise (no torch matmul in forward).
        if not TRITON_AVAILABLE or not (A.is_cuda and B.is_cuda):
            raise RuntimeError("This implementation requires Triton and CUDA tensors to run.")

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor in fp32 for stable accumulation; host can cast after kernel if needed
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float32)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Choose tile sizes; these work well across small and large dims
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_b, BLOCK_N))

        # Launch Triton kernel
        matmul_bt_kernel[grid](
            A, B, C,
            M, N_b, K_a,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Cast to original input dtype to match typical behavior (inputs are fp16 in get_inputs)
        # If you want to keep fp32 for higher precision, you can return C directly.
        # Here we match the dtype of A to align with the original model's output dtype.
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
