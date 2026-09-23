import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Single optimized Triton GEMM: computes C = A @ B.T
# A: [M, K], B: [N, K] (original), output C: [M, N]
@triton.jit
def matmul_bt_tuned_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,     # A strides: row (M), col (K)
    stride_bn, stride_bk,     # B strides: row (N), col (K)
    stride_cm, stride_cn,     # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a_vals = a_vals.to(tl.float32)

        # Load B tile as if transposed: [BLOCK_K, BLOCK_N]
        # B[k, n] = B_ptr + n * stride_bn + k * stride_bk
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + k_ids[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (k_ids[:, None] < K)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b_vals = b_vals.to(tl.float32)

        # Accumulate
        acc += tl.dot(a_vals, b_vals)

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
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
            raise ValueError(f"Cannot multiply A[M, K_a]={A.shape} with B[N_b, K_b]={B.shape}: K dimensions must match")

        # Triton and CUDA must be available; enforce Triton-only requirement (no torch matmul in forward)
        if not TRITON_AVAILABLE or not (A.is_cuda and B.is_cuda):
            raise RuntimeError("This implementation requires Triton and CUDA tensors to run.")

        # Ensure contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor: fp16 to match typical input dtype; accumulation in kernel is fp32
        C = torch.empty((M, N_b), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bk = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Choose tile sizes and num_warps based on M for performance
        if M <= 64:
            BLOCK_M = 32
            BLOCK_N = 128
            BLOCK_K = 64
            num_warps = 4
        elif M <= 256:
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            num_warps = 4
        else:
            BLOCK_M = 128
            BLOCK_N = 128
            BLOCK_K = 128
            num_warps = 8

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_b, BLOCK_N))

        # Launch the single optimized Triton kernel
        matmul_bt_tuned_kernel[grid](
            A, B, C,
            M, N_b, K_a,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=2  # modest pipelining; can be tuned further
        )

        return C


def run(*args):
    return ModelNew()(*args)
