import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes C = A @ B.T
# A: [M, K], B: [N, K] (original layout), output C: [M, N] (float16)
@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # A strides: row (M), col (K)
    stride_bn, stride_bk,     # B strides: row (N), col (K) in original B
    stride_cm, stride_cn,     # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr,    # tile size in M
    BLOCK_N: tl.constexpr,    # tile size in N
    BLOCK_K: tl.constexpr     # tile size in K
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[k, n] where A is [M, K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile as transposed: B[k, n] from original B[n, k]
        # For each k, n, element is B_ptr + n*stride_bn + k*stride_bk
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += A_tile @ B_tile
        acc += tl.dot(a, b)

    # Store result to C as float16
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        if not (A.ndim == 2 and B.ndim == 2):
            raise ValueError("A and B must be 2D tensors")
        M, K_a = A.shape
        N_b, K_b = B.shape
        if K_a != K_b:
            raise ValueError(f"Cannot multiply A[M, K_a]={A.shape} with B[N_b, K_b]={B.shape}: K dimensions must match (K_a == K_b)")

        # Triton and CUDA must be available; otherwise, raise to enforce Triton-only requirement.
        if not TRITON_AVAILABLE or not (A.is_cuda and B.is_cuda):
            raise RuntimeError("This implementation requires Triton and CUDA tensors to run.")

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor in fp16 to match typical input dtype
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Launch Triton kernel over tiles of M and N
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_b, BLOCK_N))

        matmul_bt_kernel[grid](
            A, B, C,
            M, N_b, K_a,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return C


def run(*args):
    return ModelNew()(*args)
