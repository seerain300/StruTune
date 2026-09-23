import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T, where A: [M, K], B: [K, N], output C: [M, N]
# Each program computes one output element (m, n) and loops over K:
# C[m, n] = sum_{k=0}^{K-1} A[m, k] * B[n, k]  (since B^T[k, n] = B[n, k])
@triton.jit
def matmul_transB_per_element(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,
):
    # 2D grid: one program per (m, n)
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K and accumulate
    for k in range(0, K):
        a = tl.load(A_ptr + m * stride_am + k * stride_ak)
        b = tl.load(B_ptr + n * stride_bn + k * stride_bk)
        acc += a.to(tl.float32) * b.to(tl.float32)

    # Store as float16 to match original dtype (get_inputs uses float16)
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

    # Output tensor (float16 to match provided get_inputs)
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Strides (in elements)
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Grid covers all M and N
    grid = (M, N)

    matmul_transB_per_element[grid](
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
        # Replace torch.matmul(A, B.T) with Triton computation
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
