import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # for A [M, K], contiguous: stride_am=K, stride_ak=1
    stride_bk, stride_bn,   # for B [K, N], contiguous: stride_bk=N, stride_bn=1
    stride_cm, stride_cn,   # for C [M, N], contiguous: stride_cm=N, stride_cn=1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: A[m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile as B_T[k, n] = B[n, k] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result C[m, n] = acc, cast to FP16 to match output dtype
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. A: [M, K], B: [K, N], returns C: [M, N].
    """
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("A and B must be CUDA tensors for Triton kernel.")
    # Ensure contiguous for predictable strides
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise RuntimeError(f"Incompatible shapes: A is {A.shape}, B is {B.shape}. Expected A[*, K], B[K, *].")

    # Output tensor, match input dtype (FP16)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides (in elements)
    stride_am, stride_ak = A.stride(0), A.stride(1)  # contiguous: stride_am=K, stride_ak=1
    stride_bk, stride_bn = B.stride(0), B.stride(1)  # contiguous: stride_bk=N, stride_bn=1
    stride_cm, stride_cn = C.stride(0), C.stride(1)  # contiguous: stride_cm=N, stride_cn=1

    # Launch configuration: conservative and previously correct
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
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
        # Triton-based matmul(A, B.T)
        return _triton_matmul_transpose(A, B)


def run(*args):
    return ModelNew()(*args)
