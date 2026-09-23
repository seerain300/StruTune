import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T, elementwise
# A: [M, K], B: [K, N], C: [M, N]
@triton.jit
def matmul_transB_elem_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    # Each program handles one output element C[m, n]
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K dimension and accumulate
    for k in range(0, K):
        A_val = tl.load(A_ptr + m * stride_am + k * stride_ak)
        B_val = tl.load(B_ptr + n * stride_bn + k * stride_bk)
        acc += A_val.to(tl.float32) * B_val.to(tl.float32)

    # Store result as float16 (matching get_inputs dtype)
    tl.store(C_ptr + m * stride_cm + n * stride_cn, acc.to(tl.float16))

def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton, A: [M, K], B: [K, N], output: [M, N].
    """
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

    # Ensure CUDA tensors
    if not A.is_cuda:
        A = A.cuda(non_blocking=True)
    if not B.is_cuda:
        B = B.cuda(non_blocking=True)

    # Make contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor: float16 to match provided get_inputs
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides in elements
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Launch 2D grid: one program per output element
    grid = (M, N)
    matmul_transB_elem_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        num_warps=1, num_stages=1,
    )
    return C

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are provided
        if len(A) == 0 or len(B) == 0:
            raise ValueError("Inputs A and B must be provided")
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
