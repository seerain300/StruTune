# task: flashinfer/gemm_n4096_k4096
# batch: stts3turn
# pass_at_1: 0.75
# final_geomean_speedup(A800, official re-eval): 0.430
import torch
import triton
import triton.language as tl

# Autotuned matmul kernel: computes C[M, N] = A[M, K] @ B_T[K, N]
# We pass B_T by using B with swapped strides: B_T[k, n] = B[n, k]
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),

        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),

        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=16, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=5),
    ],
    key=["M", "N", "K"],  # choose best config per problem size
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # these define B_T: k-axis stride and n-axis stride
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in BLOCK_K chunks
    offs_k_init = tl.arange(0, BLOCK_K)
    for k in range(0, K, BLOCK_K):
        offs_k = k + offs_k_init

        # A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B_T tile: shape [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C (fp16)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects exactly two tensors: A and B.")
        A, B = args

        # Device checks
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew requires CUDA tensors. Please move inputs to CUDA.")

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A is [M, K], B is [N, K] to emulate B.T without materializing it
        M, K = A.shape
        if B.dim() != 2:
            raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}.")
        N, K2 = B.shape
        if K2 != K:
            raise ValueError(f"B must have second dim equal to A.shape[1] (K). Got K={K2}, expected {K}.")

        # Output tensor (fp16 as in the provided example)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B_T strides: k-axis stride is B.stride(1), n-axis stride is B.stride(0)
        stride_bk = B.stride(1)  # corresponds to original B's second dim
        stride_bn = B.stride(0)  # corresponds to original B's first dim
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: depends on tile sizes chosen by autotune; Triton will pass META with BLOCK_M/BLOCK_N
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        _matmul_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
