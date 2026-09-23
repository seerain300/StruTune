import torch
import triton
import triton.language as tl


@triton.jit
def row_matvec_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,    # A: [M, K]
    stride_bk, stride_bn,    # B: [K, N]
    stride_cm, stride_cn,    # C: [M, N]
):
    # Each program handles one row m and one tile of columns n
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Guard in case grid is larger than M (not used here, but safe)
    if pid_m >= M:
        return

    # Column offsets for this tile
    n_offsets = pid_n * 64 + tl.arange(0, 64)  # tile size 64; can adjust to 128 later

    # Accumulator for this tile [1, BLOCK_N]
    acc = tl.zeros((1, 64), dtype=tl.float32)

    # Loop over k dimension in chunks of 64 for vectorized loads
    for k0 in range(0, K, 64):
        # Load A[m, k0:k0+64] -> vector a of length 64
        a_ptrs = A_ptr + pid_m * stride_am + (k0 + tl.arange(0, 64)) * stride_ak
        a_mask = (k0 + tl.arange(0, 64)) < K
        a_vec = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype follows A_ptr
        a32 = a_vec.to(tl.float32)  # cast to fp32 for accumulation

        # Load B[n_offsets, k0:k0+64] -> matrix b_tile of shape [64, BLOCK_N]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + (k0 + tl.arange(0, 64))[:, None] * stride_bk
        b_mask = ((k0 + tl.arange(0, 64))[:, None] < K) & (n_offsets[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype follows B_ptr
        b32 = b_tile.to(tl.float32)

        # Accumulate: acc += a_vec[:, None] * b_tile
        # a_vec is [64], b_tile is [64, 64], result is [64, 64]
        prod = a32[:, None] * b32  # broadcast a across columns
        acc += prod  # acc has shape [1, 64], we sum contributions over k

    # Store the result into C[m, n_offsets] as fp16
    c_ptrs = C_ptr + pid_m * stride_cm + n_offsets * stride_cn
    c_mask = n_offsets < N
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton, with A: [M, K], B: [K, N], returns C: [M, N].
    All computation is done in Triton. A and B must be CUDA tensors.
    """
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors for Triton."
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors."

    # Ensure contiguous tensors for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Incompatible shapes: A is {A.shape}, B is {K_b}x{N}"

    # Output tensor in fp16; accumulation in fp32 inside kernel
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Grid: one program per (row m, column tile)
    BLOCK_N = 64
    grid = (M, triton.cdiv(N, BLOCK_N))

    row_matvec_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # ModelNew replaces the original run(A, B) with Triton-based matmul(A, B.T)
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors (evaluation harness should provide CUDA tensors)
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        # Triton computation: C = A @ B.T
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
