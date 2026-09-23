import torch
import triton
import triton.language as tl

# Generic matmul kernel: computes C[M, N] = A[M, K] @ B_T[K, N]
# We pass B_T by using B with swapped strides: B_T[k, n] = B[n, k]
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # these define B_T: k-axis stride and n-axis stride
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for 2D launch
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k = 0
    while k < K:
        offs_k = k + offs_k_init

        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles, cast to float32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

        k += BLOCK_K

    # Write back to C in original dtype (fp16 as per input example)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Cast back to fp16 for storage (original tensors are fp16)
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two input tensors: A and B.")
        A, B = args

        # Ensure CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew requires CUDA tensors. Please move inputs to CUDA.")

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        N = B.shape[0]  # since B is [N, O] and we will use strides to treat it as transposed
        # For matmul A[M,K] @ B_T[K,N], B_T[k,n] = B[n,k], so B must have shape [N, K]
        if B.dim() != 2 or B.shape[1] != K:
            raise ValueError(f"B must be 2D with shape [N, K], got {tuple(B.shape)} and K={K}.")

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B_T strides: k-axis stride is B.stride(1), n-axis stride is B.stride(0)
        stride_bk = B.stride(1)  # corresponds to original B's second dim
        stride_bn = B.stride(0)  # corresponds to original B's first dim
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters (tuned for fp16; can be adjusted)
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_kernel[grid](
            A, B, C,
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
