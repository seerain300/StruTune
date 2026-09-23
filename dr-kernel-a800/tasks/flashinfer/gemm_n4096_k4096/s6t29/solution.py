import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # BT is [N, K]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling across M and N. With BLOCK_M=1 and BLOCK_N=1, each program handles one element.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    m = pid_m  # since BLOCK_M=1
    n = pid_n  # since BLOCK_N=1

    # Accumulator as float32 for numerical stability
    acc = 0.0  # scalar

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Sum contributions from each kk in the chunk
        for kk in range(0, BLOCK_K):
            k = k0 + kk
            # Masks for bounds
            a_mask = m < M and k < K
            b_mask = n < N and k < K
            # Load A[m, k]
            a_val = tl.load(A_ptr + m * stride_am + k * stride_ak, mask=a_mask, other=0.0)
            # Load BT[n, k] (BT is [N, K])
            b_val = tl.load(BT_ptr + n * stride_bn + k * stride_bk, mask=b_mask, other=0.0)
            # Accumulate (casts to float32 internally)
            acc += a_val * b_val

    # Store result to C[m, n]
    store_mask = m < M and n < N
    tl.store(C_ptr + m * stride_cm + n * stride_cn, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D and contiguous
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Create B_T = B.t() so kernel can index BT[n, k]
        BT = B.t().contiguous()  # BT shape: [N, K]

        # Allocate output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = BT.stride(0)  # along N
        stride_bk = BT.stride(1)  # along K
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid across M and N, one program per element
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 32

        grid = (M, N)

        # Launch kernel
        matmul_AT_kernel[grid](
            A, BT, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
