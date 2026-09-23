import torch
import triton
import triton.language as tl


@triton.jit
def outer_product_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,      # A strides for [M, K]
    stride_btk, stride_btn,    # BT strides for [N, K] (BT = B.T)
    stride_cm, stride_cn,      # C strides for [M, N]
):
    # Each program computes one output element c[m, n] by looping over K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # If grid is larger than M or N (shouldn't happen if we set grid to (M, N)), we guard anyway
    if pid_m >= M or pid_n >= N:
        return

    m = pid_m
    n = pid_n

    # FP32 accumulator for this single output element
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K and accumulate outer product
    for k in range(0, K):
        a_ptr = A_ptr + m * stride_am + k * stride_ak
        bt_ptr = BT_ptr + n * stride_btn + k * stride_btk
        a = tl.load(a_ptr)
        bt = tl.load(bt_ptr)
        # a and bt are scalars, promote to fp32 before multiply
        acc += (a.to(tl.float32)) * (bt.to(tl.float32))

    # Store result as fp16 (match original input dtype which is fp16 in provided get_inputs)
    c_ptr = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c_ptr, acc.to(tl.float16))


def _outer_product_triton(A: torch.Tensor, BT: torch.Tensor) -> torch.Tensor:
    # A: [M, K], BT: [N, K] (B.T), C: [M, N]
    assert A.is_cuda and BT.is_cuda, "Triton kernel requires CUDA tensors"
    M, K = A.shape
    N, K2 = BT.shape
    assert K == K2, "Incompatible shapes for A and B.T"

    # Ensure contiguous for predictable strides (optional but recommended)
    A = A.contiguous()
    BT = BT.contiguous()

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Launch grid over all output elements
    grid = (M, N)

    outer_product_kernel[grid](
        A, BT, C,
        M, N, K,
        A.stride(0), A.stride(1),
        BT.stride(0), BT.stride(1),
        C.stride(0), C.stride(1),
        num_warps=1,  # one warp per program is sufficient; simple and correct
        num_stages=1,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args

        # Ensure CUDA tensors
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)

        # Construct B_T explicitly (non-contiguous view) and use it in Triton
        BT = B.transpose(1, 0)  # shape [N, K], matches kernel expectation

        # Compute C = A @ B.T using Triton
        C = _outer_product_triton(A, BT)

        # Return result as a list (to match the original interface)
        return [C]


def run(*args):
    return ModelNew()(*args)
