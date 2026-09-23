import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K], contiguous: stride_am=K, stride_ak=1
    stride_bk, stride_bn,   # B is [K, N], contiguous: stride_bk=N, stride_bn=1
    stride_cm, stride_cn,   # C is [M, N], contiguous: stride_cm=N, stride_cn=1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: each program handles a tile [BLOCK_M x BLOCK_N] of C
    pid_m = tl.program_id(0)  # tile index along M
    pid_n = tl.program_id(1)  # tile index along N

    # Compute offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B tile: we want B_T[k, n] = B[n, k], shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C: C[m, n] = acc
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)  # acc is FP32; Triton will cast to C dtype if needed.


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T using Triton. Assumes A: [M, K], B: [K, N].
    A and B are made contiguous on host before launching the kernel.
    """
    # Ensure tensors are on CUDA and contiguous
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Triton kernel requires CUDA tensors. Move inputs to .cuda().")
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"Incompatible shapes: A is [M, {K}], B is [{Kb}, N].")

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)  # accumulate in FP32
    # Note: For storing, Triton will cast to the pointer's dtype (we allocate C in FP32).
    # If you want FP16 output to match the original, we can cast after the kernel.
    # For now, we keep FP32 to ensure numerical stability and correctness.

    # Fixed tiling that previously passed correctness; conservative and robust
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Return in FP16 to match original run()'s dtype (A and B are FP16 in the provided get_inputs)
    return C.to(torch.float16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two tensors: A and B
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors for Triton
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
