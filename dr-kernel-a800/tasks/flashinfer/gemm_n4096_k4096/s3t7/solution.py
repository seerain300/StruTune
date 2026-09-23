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

    # Loop over K in chunks of BLOCK_K (constexpr step)
    for k0 in range(0, K, BLOCK_K):
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load BT tile: shape (BLOCK_K, BLOCK_N), BT is (K, N)
        BT_ptrs = BT_ptr + ((k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(BT_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate in fp32
        acc += tl.dot(a, bt)

    # Store result C[m, n] = acc (cast to output dtype on store)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure CUDA tensors for Triton; fallback to PyTorch only if not CUDA
        if not (A.is_cuda and B.is_cuda):
            # CPU fallback for robustness (CUDA path must use Triton)
            return torch.matmul(A, B.T)

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Compute BT = B.T contiguous: shape (K, N)
        BT = B.transpose(0, 1).contiguous()

        # Shapes
        M, K = A.shape
        N = BT.shape[1]  # BT is (K, N)

        # Allocate output as float16 to match original dtype (compute in fp32, store fp16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = BT.stride(0)
        stride_bn = BT.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile sizes (conservative to avoid shared memory issues)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid over tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel; choose num_warps and num_stages to balance performance and resource usage
        matmul_at_bT_kernel[grid](
            A, BT, C,
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
