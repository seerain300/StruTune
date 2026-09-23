import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointers to the output tile
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(A_tile_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load BT tile: BT is B.T with shape (K, N); BT[k, n]
        BT_tile_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)
        b = tl.load(BT_tile_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Accumulate using dot product
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C; Triton will cast to the pointer's dtype if needed
    tl.store(C_tile_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: (M, K), B: (N, K), returns C: (M, N).
    Assumes tensors are CUDA and dtype float16.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"

    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, "Inner dimensions must match (K must be the same for A and B)"

    # Make B.T contiguous for coalesced loads
    BT = B.T.contiguous()  # shape (K, N)

    # Output tensor with same dtype as A
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Conservative tiling that passed correctness across all workloads
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_at_bT_kernel[grid](
        A, BT, C,
        M, N, K,
        A.stride(0), A.stride(1),
        BT.stride(0), BT.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,  # slightly higher warps can improve throughput on modern GPUs
        num_stages=3,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect 2 tensors: A and B
        A, B = args
        # If not CUDA, fall back to PyTorch
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A, B.T)
        # Use Triton for computation
        return run_triton(A, B)


# Optional helpers for local testing
def get_inputs():
    # Use CUDA tensors to trigger Triton path in local tests
    A = torch.randn([1, 4096], dtype=torch.float16, device='cuda')
    B = torch.randn([4096, 4096], dtype=torch.float16, device='cuda')
    return [A, B]


def fused_operator(tensor_0, tensor_1):
    _out = ModelNew()(tensor_0, tensor_1)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
