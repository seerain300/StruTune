import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per (m, n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute m and n indices
    # Since BLOCK_M=1 and BLOCK_N=1, each program handles exactly one element (m=pid_m, n=pid_n)
    m = pid_m
    n = pid_n

    # Accumulator (use fp32 for numerical stability)
    acc = 0.0  # scalar

    # Loop over K dimension
    # Note: Triton supports range loops; ensure K is passed as int.
    for k in range(0, K):
        # Load A[m, k]
        a_ptr = A_ptr + m * stride_am + k * stride_ak
        a = tl.load(a_ptr)  # scalar load
        # Load B_T[n, k] which corresponds to original B[k, n]
        b_ptr = B_ptr + n * stride_bk + k * stride_bn
        b = tl.load(b_ptr)  # scalar load
        acc += a * b

    # Store to C[m, n]
    c_ptr = C_ptr + m * stride_cm + n * stride_cn
    # Since we store a scalar, mask is optional here, but we can still guard
    # In this kernel, we assume grid exactly matches (M, N), so m<M and n<N always.
    tl.store(c_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"

        # Make contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)  # for B_T index (n, k) == B[k, n], this is B's row stride along k
        stride_bn = B.stride(1)  # for B_T index (n, k) == B[k, n], this is B's col stride along n
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: one program per (m, n)
        grid = (M, N)

        # Launch kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=1, BLOCK_N=1, BLOCK_K=1,
            num_warps=1, num_stages=1,
        )

        return C


def run(*args):
    return ModelNew()(*args)
