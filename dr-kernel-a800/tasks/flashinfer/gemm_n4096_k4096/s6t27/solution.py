import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,   # BT_ptr has shape [N, K] conceptually
    stride_cm, stride_cn,
):
    # 2D program ids: one program per (m, n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute indices for this program
    m_idx = pid_m
    n_idx = pid_n

    # Accumulator in float32
    acc = tl.zeros((), dtype=tl.float32)

    # Sum over K: C[m, n] = sum_k A[m, k] * BT[n, k]
    # Directly loop over k from 0 to K-1
    # Note: In Triton, Python 'range' works with runtime K here as well.
    for k in range(0, K):
        # Load A[m, k]
        a = tl.load(A_ptr + m_idx * stride_am + k * stride_ak)
        # Load BT[n, k]
        b = tl.load(BT_ptr + n_idx * stride_bn + k * stride_bk)
        # Accumulate in float32
        acc += a.to(tl.float32) * b.to(tl.float32)

    # Store result to C[m, n] as float16 (matching original get_inputs dtype)
    tl.store(C_ptr + m_idx * stride_cm + n_idx * stride_cn, acc.to(tl.float16))


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Create B_t as a contiguous transposed view: shape [N, K]
        B_t = B.t().contiguous()

        # Output tensor, same dtype as inputs (float16 in provided example)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B_t.stride(0)  # corresponds to N dimension in BT
        stride_bk = B_t.stride(1)  # corresponds to K dimension in BT
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: one program per (m, n)
        grid = (M, N)

        # Launch Triton kernel
        matmul_AT_kernel[grid](
            A, B_t, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            num_warps=1,  # minimal for robust coverage of tiny grids
            num_stages=1,
        )

        return C


def run(*args):
    return ModelNew()(*args)
