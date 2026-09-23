import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_flat_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: shape [M, K]
    stride_bk, stride_bn,   # strides for B: shape [K, N]
    stride_cm, stride_cn,   # strides for C: shape [M, N]
    L,                      # total number of output elements (M * N)
    BLOCK: tl.constexpr,
):
    # 1D grid: each program handles a block of output elements
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_out = offs < L

    # Compute (m, n) for each output index offs
    m = offs // N
    n = offs % N

    # Accumulator for each output (FP32)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    # Reduction over K
    # For each k, load A[m, k] (vector of length BLOCK) and B[n, k] (vector of length BLOCK),
    # then acc += A * B (elementwise product for each output lane).
    for k in range(0, K):
        # Load A[m, k]: shape [BLOCK], masked by valid m and k range (m always valid here since m<M, but we keep it general)
        a_ptrs = A_ptr + m * stride_am + k * stride_ak
        # Note: m is a vector; Triton supports vector indexing in pointer arithmetic. The mask ensures we don't read invalid m.
        a_mask = mask_out
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T[k, n] = B[n, k]: shape [BLOCK], masked by valid n and k
        b_ptrs = B_ptr + n * stride_bn + k * stride_bk
        b_mask = mask_out
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += a.to(tl.float32) * b.to(tl.float32)

    # Store results to C at positions (m, n)
    # We need to compute per-lane pointers for C[m, n] using strides.
    c_ptrs = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_out)


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], C: [M, N].
    """
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M, K]={K}, B is [Kb, N]={Kb} vs {N}"

    # Ensure contiguous CUDA tensors
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor: match dtype with inputs (float16 in provided get_inputs)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # 1D grid over flattened output
    L = M * N
    BLOCK = 1024  # one program handles 1024 outputs; small K loop ensures reasonable runtime
    grid = (triton.cdiv(L, BLOCK),)

    # Launch kernel; pass strides in elements (torch.stride returns elements)
    matmul_transpose_flat_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),     # strides for A: [M, K]
        B.stride(0), B.stride(1),     # strides for B: [K, N]
        C.stride(0), C.stride(1),     # strides for C: [M, N]
        L,
        BLOCK=BLOCK,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Compute A @ B.T to match original run behavior
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
