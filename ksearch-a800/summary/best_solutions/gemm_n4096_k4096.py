# task: gemm_n4096_k4096
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=43/43 geomean=0.839x
# feedback best (5-workload sample during search): 0.840x
# torch fallback audit: A·核心靠库 (matmul)
# tokens: 853,791

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_m_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, 4096, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None],
            other=0.0,
        )
        b = tl.load(b_ptrs)

        acc += tl.dot(a, tl.trans(b))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=mask_m[:, None],
    )


def run(A, B):
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional tensors")
    if A.shape[1] != 4096:
        raise ValueError("A must have shape [M, 4096]")
    if B.shape != (4096, 4096):
        raise ValueError("B must have shape [4096, 4096]")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError("A and B must have dtype torch.float16")

    if not torch.cuda.is_available():
        if A.is_cuda or B.is_cuda:
            raise RuntimeError("CUDA tensors were provided, but CUDA is unavailable")
        raise RuntimeError("CUDA is required to execute the GEMM")

    output_device = A.device

    if A.is_cuda:
        execution_device = A.device
    elif B.is_cuda:
        execution_device = B.device
    else:
        execution_device = torch.device("cuda", torch.cuda.current_device())

    with torch.cuda.device(execution_device):
        if A.device == execution_device:
            A_gpu = A
        elif A.is_cuda:
            A_gpu = A.to(device=execution_device)
        else:
            A_gpu = A.cuda(device=execution_device)

        if B.device == execution_device:
            B_gpu = B
        elif B.is_cuda:
            B_gpu = B.to(device=execution_device)
        else:
            B_gpu = B.cuda(device=execution_device)

        M = A_gpu.shape[0]

        if M == 0:
            C_gpu = torch.empty(
                (0, 4096),
                device=execution_device,
                dtype=torch.float16,
            )
        elif M <= 16:
            C_gpu = torch.empty(
                (M, 4096),
                device=execution_device,
                dtype=torch.float16,
            )
            _compact_m_gemm_kernel[(128,)](
                A_gpu,
                B_gpu,
                C_gpu,
                M,
                A_gpu.stride(0),
                A_gpu.stride(1),
                B_gpu.stride(0),
                B_gpu.stride(1),
                C_gpu.stride(0),
                C_gpu.stride(1),
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=128,
                num_warps=4,
                num_stages=4,
            )
        else:
            C_gpu = torch.matmul(A_gpu, B_gpu.transpose(0, 1))

    if output_device != execution_device:
        return C_gpu.to(device=output_device)
    return C_gpu