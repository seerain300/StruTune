import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bT_kernel_fp16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp16 (output dtype matches A/B dtype which is fp16 in harness)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # Load A tile: shape [BLOCK_M, BLOCK_K], A is fp16
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile: B has shape [K, N], we want [BLOCK_K, BLOCK_N]
        # Index B[k, n] as k*stride_bk + n*stride_bn (fp16)
        b_ptrs = B_ptr + k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp16 (Triton will perform fp16 dot if both inputs are fp16)
        # Using tl.dot; for fp16, it performs the multiply-accumulate in lower precision,
        # which closely matches PyTorch's fp16 matmul behavior.
        acc += tl.dot(a, b)

    # Store result to C (fp16 output)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]")

        # Output tensor: same dtype as A (fp16 in harness)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Tile sizes: robust default for many shapes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid over tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_bT_kernel_fp16[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return C


def run(*args):
    return ModelNew()(*args)
