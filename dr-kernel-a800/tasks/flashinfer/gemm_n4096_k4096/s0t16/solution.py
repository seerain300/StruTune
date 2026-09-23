import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_general_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # for B_T: B_T[k, n] = B[n, k]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[offs_m, offs_k]
        A_ptrs = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # dtype follows A (fp16/bf16/fp32)

        # Load B_T tile: B_T[offs_k, offs_n] = B[offs_n, offs_k]
        B_ptrs = B + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)  # dtype follows B

        # Accumulate in fp32
        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))

    # Store to C (cast to C's dtype)
    C_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Cast accumulator to C dtype (assume C is same dtype we want to produce)
    # Triton allows us to store fp32 values into a fp16/ bf16/ fp32 tensor, but we explicitly cast.
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Inputs must be 2D and CUDA
        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D tensors"
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device"

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N = B.shape[0]  # B is [N, K]; we index as B_T[k, n] = B[n, k]

        # Output dtype: match PyTorch's result_type of A @ B.T == result_type(A, B)
        out_dtype = torch.result_type(A, B)
        C = torch.empty((M, N), device=A.device, dtype=out_dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(1)  # B's second dim (K)
        stride_bn = B.stride(0)  # B's first dim (N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # 2D grid over tiles
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        matmul_general_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C


def run(*args):
    return ModelNew()(*args)
