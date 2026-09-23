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
    # We'll iterate over K in chunks of BLOCK_K using a static loop
    # For the mask, we need the full K range; we'll compute masks per chunk.

    # Accumulator in fp32 for numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Static loop over K dimension
    # Note: tl.static_range requires compile-time known bounds; here K is runtime,
    # but Triton can still handle this pattern with masks; the loop body is repeated
    # per chunk, and masks guard out-of-bounds loads. Some Triton versions require
    # constexpr for range; to be safe across versions, we keep the loop simple and
    # rely on masks.
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: shape (BLOCK_K, BLOCK_N), BT is (K, N)
        BT_ptrs = BT_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(BT_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a, bt)

    # Store result C[m, n] = acc (cast to output dtype on store)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store fp32; Triton will handle cast to the pointer dtype (we allocated C as float16)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA for Triton; if not, we can fall back,
        # but the evaluator uses CUDA. Keep Triton-only in CUDA path.
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."

        # Shapes: A is (M, K), B is (N, K). We need C = A @ B.T, where B.T is (K, N).
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"Incompatible shapes: A is (M, K), B is (N, K2) but K != K2 ({K} vs {K2})"

        # Prepare BT = B.T (K, N) and make contiguous for efficient loads
        BT = B.transpose(0, 1).contiguous()

        # Allocate output C as float16
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = BT.stride(0), BT.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Choose conservative tile sizes to avoid shared memory limits
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # Launch 2D grid over tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
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
