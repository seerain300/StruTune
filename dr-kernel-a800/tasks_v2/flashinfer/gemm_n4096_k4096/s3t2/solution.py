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

        # Accumulate
        acc += tl.dot(a, bt)

    # Store result C[m, n] = acc
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)  # acc is fp32; Triton will cast to C's dtype (float16)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Validate shapes
        if A.dim() != 2:
            raise RuntimeError(f"A must be 2D, got shape {tuple(A.shape)}")
        if B.dim() != 2:
            raise RuntimeError(f"B must be 2D, got shape {tuple(B.shape)}")

        M, K = A.shape
        N, K2 = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is (M, K)={A.shape}, B is (N, K2)={B.shape}, K must match")

        # Ensure CUDA tensors; Triton-only path
        if not A.is_cuda:
            raise RuntimeError("Input A must be on CUDA for Triton execution")
        if not B.is_cuda:
            raise RuntimeError("Input B must be on CUDA for Triton execution")

        # Prepare BT = B.transpose(0, 1) -> shape (K, N), make contiguous for performance
        BT = B.transpose(0, 1).contiguous()

        # Allocate output C (M, N) as float16 to match original dtype behavior
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Ensure inputs are contiguous
        A = A.contiguous()
        BT = BT.contiguous()
        # C is allocated contiguous

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = BT.stride(0)  # BT is (K, N)
        stride_bn = BT.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid over tiles
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

        # Launch Triton kernel (no torch matmul in CUDA path)
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        return C


def run(*args):
    return ModelNew()(*args)
