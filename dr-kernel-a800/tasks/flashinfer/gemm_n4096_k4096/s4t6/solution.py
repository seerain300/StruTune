import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_simple_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,   # C is [M, N]
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row m and a block of columns n
    pid_m = tl.program_id(0)   # row index
    pid_n = tl.program_id(1)   # block index along columns

    # Column offsets for this block
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # FP32 accumulator for this row block
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension
    # For each k, accumulate A[m, k] * B[n, k] into acc
    for k in range(0, K):
        # Load A[m, k] (scalar). Grid ensures pid_m < M.
        a_val = tl.load(A_ptr + pid_m * stride_am + k * stride_ak)
        # Load B[n, k] as a vector
        b_vec = tl.load(B_ptr + k * stride_bk + n_offsets * stride_bn, mask=n_mask, other=0.0)
        # Accumulate in FP32
        acc += a_val.to(tl.float32) * b_vec.to(tl.float32)

    # Store result to C[m, n_offsets] as FP16
    tl.store(C_ptr + pid_m * stride_cm + n_offsets * stride_cn, acc, mask=n_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], C: [M, N].
    Returns C as float16 tensor (matching typical inputs).
    """
    # Require CUDA tensors (evaluation harness provides CUDA tensors).
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Triton kernel requires CUDA tensors; ensure inputs are on CUDA.")
    # Ensure contiguity for simple stride arithmetic
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"A's second dim {K} must match B's first dim {Kb}")

    # Output C in FP16
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides (in elements)
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Launch configuration: one program per row and per column block
    BLOCK_N = 128
    grid = (M, triton.cdiv(N, BLOCK_N))

    matmul_transpose_simple_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original Model.forward computes run(A, B) which is A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Do not move devices; rely on the harness to provide CUDA tensors.
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
