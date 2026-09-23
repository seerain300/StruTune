import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: one program per output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B tiles
    # A: [M, K], strides (stride_am, stride_ak)
    # B: [K, N], strides (stride_bk, stride_bn); we load B[k, n] using its strides to emulate B.T
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Cast to fp32 for stable accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

        # Advance pointers for next K-block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store results
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: [M, K], B: [K, N]
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N]"

        # Ensure contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        # Compute in fp32, output in fp16 to match evaluator
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Fixed block sizes chosen for robustness
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel (all computation in Triton)
        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast to fp16 to match reference output
        return C.to(torch.float16)


def run(*args):
    return ModelNew()(*args)
