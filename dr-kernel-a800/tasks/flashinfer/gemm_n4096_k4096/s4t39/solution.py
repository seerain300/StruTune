import torch
import triton
import triton.language as tl


@triton.jit
def outer_product_kernel_mn(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: [M, K]
    stride_bn, stride_bk,   # strides for B_T: [N, K]
    stride_cm, stride_cn,   # strides for C: [M, N]
):
    # Grid is 2D over (m, n). Each program computes one element c[m, n]
    m = tl.program_id(0)  # row index in C
    n = tl.program_id(1)  # col index in C

    # If out of bounds (shouldn't happen with proper grid), return
    if m >= M or n >= N:
        return

    # Accumulator in FP32
    acc = 0.0

    # Loop over K, accumulate a[m, k] * BT[n, k]
    for k in range(0, K):
        # Load a[m, k]
        a_val = tl.load(A_ptr + m * stride_am + k * stride_ak)
        # Load BT[n, k] where BT is [N, K]
        b_val = tl.load(BT_ptr + n * stride_bn + k * stride_bk)
        # Accumulate in FP32
        acc += a_val.to(tl.float32) * b_val.to(tl.float32)

    # Store result to C[m, n] as FP16
    tl.store(C_ptr + m * stride_cm + n * stride_cn, acc.to(tl.float16))


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton, with explicit B_T = B.transpose(1, 0).
    A: [M, K], B: [K, N], C: [M, N]. All tensors on CUDA, float16.
    """
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Inputs must be CUDA tensors for Triton.")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise RuntimeError("This Triton implementation expects float16 inputs.")

    # Ensure contiguity for simple stride handling
    A = A.contiguous()
    B = B.contiguous()

    # Explicitly construct B_T as [N, K]
    BT = B.transpose(1, 0).contiguous()

    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, "B's first dimension must equal A's second dimension (K)."

    # Output tensor C [M, N], float16
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bn, stride_bk = BT.stride(0), BT.stride(1)  # BT is [N, K]
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Launch Triton: one program per (m, n)
    grid = (M, N)
    outer_product_kernel_mn[grid](
        A, BT, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        num_warps=1,  # simple per-element kernel
        num_stages=1,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two inputs: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        # Run Triton kernel
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
