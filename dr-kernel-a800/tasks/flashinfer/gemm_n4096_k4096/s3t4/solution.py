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
        # Ensure tensors are on CUDA; evaluation harness uses CUDA tensors.
        if not A.is_cuda or not B.is_cuda:
            # CPU fallback for robustness (not used in eval), but kept for correctness.
            return torch.matmul(A, B.T)

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Prepare BT = B.T (shape: (K, N)) from B which is (N, K)
        BT = B.transpose(0, 1).contiguous()

        # Output tensor
        M, K = A.shape
        Kb, N_ = BT.shape
        assert Kb == K, f"B's second dim {Kb} must equal A's second dim {K}"
        C = torch.empty((M, N_), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = BT.stride(0)
        stride_bn = BT.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Conservative block sizes to avoid shared memory overflow
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        num_warps, num_stages = 4, 2

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N_, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        return C


def run(*args):
    return ModelNew()(*args)
