import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_row_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: shape [M, K]
    stride_bk, stride_bn,   # strides for B: shape [K, N]
    stride_cm, stride_cn,   # strides for C: shape [M, N]
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row m and a block of columns of size BLOCK_N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this row block (FP32 for numerical stability)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K):
        # Load A[m, k] as scalar
        a_ptr = A_ptr + m * stride_am + k * stride_ak
        a_val = tl.load(a_ptr)
        a_val = a_val.to(tl.float32)

        # Load B[n, k] as vector over n_offsets
        b_ptrs = B_ptr + n_offsets * stride_bn + k * stride_bk
        b_mask = n_offsets < N
        b_vec = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b_vec = b_vec.to(tl.float32)

        # FMA accumulate
        acc += a_val * b_vec

    # Store results to C[m, n_offsets]
    c_ptrs = C_ptr + m * stride_cm + n_offsets * stride_cn
    c_mask = n_offsets < N
    # Cast to output dtype (FP16, matching original get_inputs)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect A and B as inputs
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args

        # Ensure CUDA tensors and contiguity for correct stride handling
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is [M, {K}], B is [{Kb}, {N}].")

        # Output tensor (FP16, matching inputs from get_inputs)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides for contiguous tensors
        stride_am = A.stride(0)  # typically K
        stride_ak = A.stride(1)  # typically 1
        stride_bk = B.stride(0)  # typically N
        stride_bn = B.stride(1)  # typically 1
        stride_cm = C.stride(0)  # typically N
        stride_cn = C.stride(1)  # typically 1

        # Launch grid: one program per row m and per column block
        BLOCK_N = 128
        grid = (M, triton.cdiv(N, BLOCK_N))

        matmul_transpose_row_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
