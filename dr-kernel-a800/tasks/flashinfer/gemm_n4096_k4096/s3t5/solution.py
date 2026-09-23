import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: A is (M, K)
    stride_bk, stride_bn,       # BT strides: BT is (K, N)
    stride_cm, stride_cn,       # C strides: C is (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids over tiles of output C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Accumulator in fp32 for numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: shape (BLOCK_K, BLOCK_N), BT is (K, N)
        BT_ptrs = BT_ptr + ((k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(BT_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a, bt)

    # Store result C[m, n] = acc (cast to output dtype on store)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are CUDA tensors and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        # Compute BT = B.T (shape becomes (K, N))
        BT = B.transpose(0, 1).contiguous()

        # Shapes
        M, K = A.shape
        Kb, N = BT.shape
        assert Kb == K, f"Internal inconsistency: B shape {B.shape}, expected K columns equal to A's K."

        # Output tensor (M, N), float16 to match example
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides (in elements)
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = BT.stride(0), BT.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Dynamically choose block sizes based on M and N to balance performance and resource usage.
        # Keep shared memory per program under ~196 KB: BLOCK_M * BLOCK_N * (BLOCK_K/2) * 4 bytes < 196608.
        if M <= 32 or N <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
            num_warps, num_stages = 2, 2
        elif M <= 128 and N <= 128:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            num_warps, num_stages = 4, 3
        else:
            # Favor larger tile along M while keeping N moderate
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
            num_warps, num_stages = 8, 4

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return C


def run(*args):
    return ModelNew()(*args)
