import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T, where
# A: [M, K] (row-major: stride_am, stride_ak)
# B: [N, P] (row-major: stride_bn, stride_bp); we will logically use B.T[k, n] = B[n, k]
# C: [M, P] (row-major: stride_cm, stride_cp)
# We accumulate in float32 for numerical stability.
@triton.jit
def matmul_btrans_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bp,
    stride_cm, stride_cp,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[offs_m, offs_k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # Masks for A load
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # Cast to fp32 for accumulation
        a = a.to(tl.float32)

        # Pointers for B^T tile: B^T[offs_k, offs_n] = B[offs_n, offs_k]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bp)
        # Masks for B load
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b = b.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back to C: C[offs_m, offs_n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cp)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two tensors: A and B
        # A: [M, K], B: [N, P]; we compute C = A @ B.T with output [M, P]
        # In the provided setup, A is [M, 4096], B is [4096, 4096], output is [M, 4096]
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B")
        A, B = args

        # Shapes
        M = A.shape[0]
        K = A.shape[1]
        N = B.shape[0]  # rows of B
        P = B.shape[1]  # cols of B; B.T has shape [P, N]

        # Allocate output as float32 for accumulation
        # We will return casted to A.dtype to match original behavior (fp16 in provided inputs)
        C = torch.empty((M, P), dtype=torch.float32, device=A.device)

        # Strides (row-major expected, but we use actual strides)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bn, stride_bp = B.stride(0), B.stride(1)
        stride_cm, stride_cp = C.stride(0), C.stride(1)

        # Tile sizes (tuned for fp16 matmul; fp32 accumulation)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(P, BLOCK_N))

        # Launch Triton kernel
        matmul_btrans_kernel[grid](
            A, B, C,
            M, P, K,  # use P as second dimension of output
            stride_am, stride_ak,
            stride_bn, stride_bp,
            stride_cm, stride_cp,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Cast back to input dtype to match original (A is fp16 in provided inputs)
        return C.to(A.dtype)