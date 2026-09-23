# solution=GPT-5.6-Sol_gemm_n4096_k4096_triton_optimized_r29 score=0.4972351299327574 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(4096, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, 4096, BLOCK_K):
        if M % BLOCK_M == 0:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=offs_m[:, None] < M,
                other=0.0,
            )
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    if M % BLOCK_M == 0:
        tl.store(c_ptrs, accumulator.to(tl.float16))
    else:
        tl.store(
            c_ptrs,
            accumulator.to(tl.float16),
            mask=offs_m[:, None] < M,
        )


def _configuration(m):
    if m <= 16:
        return 16, 64, 4, 4
    if m < 128:
        return 32, 64, 4, 4
    if m < 256:
        return 64, 64, 4, 4
    if m < 512:
        return 64, 128, 4, 4
    return 128, 128, 8, 3


def run(A, B):
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional tensors")
    if A.shape[1] != 4096 or B.shape != (4096, 4096):
        raise ValueError(
            "expected A with shape [M, 4096] and B with shape [4096, 4096]"
        )
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError("A and B must have dtype torch.float16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the Triton GEMM kernel")

    output_device = A.device

    if A.device.type == "cpu":
        A_gpu = A.cuda()
    elif A.device.type == "cuda":
        A_gpu = A
    else:
        raise ValueError(f"unsupported device for A: {A.device}")

    if B.device.type == "cpu":
        B_gpu = B.cuda(device=A_gpu.device)
    elif B.device.type == "cuda":
        B_gpu = B.to(device=A_gpu.device) if B.device != A_gpu.device else B
    else:
        raise ValueError(f"unsupported device for B: {B.device}")

    m = A_gpu.shape[0]
    C_gpu = torch.empty(
        (m, 4096),
        device=A_gpu.device,
        dtype=torch.float16,
    )

    if m != 0:
        block_m, block_n, num_warps, num_stages = _configuration(m)
        grid = (
            triton.cdiv(m, block_m) * triton.cdiv(4096, block_n),
        )

        with torch.cuda.device(A_gpu.device):
            _gemm_kernel[grid](
                A_gpu,
                B_gpu,
                C_gpu,
                A_gpu.stride(0),
                A_gpu.stride(1),
                B_gpu.stride(0),
                B_gpu.stride(1),
                C_gpu.stride(0),
                C_gpu.stride(1),
                M=m,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=64,
                GROUP_M=8,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return C_gpu.to(output_device) if output_device.type != "cuda" else C_gpu