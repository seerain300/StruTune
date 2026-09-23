import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N]
# Interpret B^T[k, n] = B[n, k] by loading B[offs_n, k] -> shape [BLOCK_N, 1] and transposing for dot.
@triton.jit
def matmul_transB_3d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Output tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-chunk start
    k0 = pid_k * BLOCK_K

    # Loop over this K chunk; use static_range so Triton can optimize
    for kk in tl.static_range(0, BLOCK_K):
        k = k0 + kk
        # Masks to guard loads/stores
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k_scalar = k < K

        # Load A_sub = A[offs_m, k] -> shape [BLOCK_M, 1]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + k * stride_ak)
        A_mask = mask_m[:, None] & mask_k_scalar
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)  # [BLOCK_M, 1]

        # Load B_sub = B_T[k, offs_n] = B[offs_n, k] -> shape [1, BLOCK_N]
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + k * stride_bk)
        B_mask = mask_n[None, :] & mask_k_scalar
        B_sub = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)  # [1, BLOCK_N]

        # Accumulate: acc += A_sub @ B_sub -> [BLOCK_M, BLOCK_N]
        acc += A_sub @ B_sub

    # Store result to C[M, N], cast to fp16 to match original dtype
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)

        # Make inputs contiguous to simplify stride handling
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        Kb, N = B.shape
        assert Kb == K, f"B's first dimension {Kb} must equal A's second dimension {K}"

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (element-wise)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)  # over K
        stride_bn = B.stride(1)  # over N
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes: tuned for general performance
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Grid covers M, N, and K fully
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
            triton.cdiv(K, BLOCK_K),
        )

        # Launch kernel
        matmul_transB_3d_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
