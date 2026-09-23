# solution=GPT-5.6-Sol_gemm_n4096_k4096_triton_optimized_r36 score=0.41837995228880115 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m_size, BLOCK_M)
    num_pid_n = tl.cdiv(4096, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 4096, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, tl.trans(b), accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=offs_m[:, None] < m_size)


def _launch_gemm(a, b):
    m_size = a.shape[0]
    output = torch.empty(
        (m_size, 4096),
        device=a.device,
        dtype=torch.float16,
    )

    if m_size <= 4:
        block_m, block_n, block_k = 4, 128, 64
        num_warps, num_stages = 4, 3
    elif m_size <= 8:
        block_m, block_n, block_k = 8, 128, 64
        num_warps, num_stages = 4, 3
    elif m_size <= 16:
        block_m, block_n, block_k = 16, 128, 64
        num_warps, num_stages = 4, 3
    elif m_size <= 32:
        block_m, block_n, block_k = 32, 128, 64
        num_warps, num_stages = 4, 3
    elif m_size <= 128:
        block_m, block_n, block_k = 32, 128, 64
        num_warps, num_stages = 4, 3
    elif m_size <= 512:
        block_m, block_n, block_k = 64, 128, 64
        num_warps, num_stages = 4, 3
    else:
        block_m, block_n, block_k = 64, 128, 64
        num_warps, num_stages = 8, 3

    grid = (
        triton.cdiv(m_size, block_m) * triton.cdiv(4096, block_n),
    )
    _gemm_kernel[grid](
        a,
        b,
        output,
        m_size,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def run(*args, **kwargs):
    if len(args) > 2:
        raise TypeError(f"run() takes 2 arguments but {len(args)} were given")

    if len(args) >= 1:
        if "A" in kwargs:
            raise TypeError("run() got multiple values for argument 'A'")
        a = args[0]
    elif "A" in kwargs:
        a = kwargs.pop("A")
    else:
        raise TypeError("run() missing required argument: 'A'")

    if len(args) >= 2:
        if "B" in kwargs:
            raise TypeError("run() got multiple values for argument 'B'")
        b = args[1]
    elif "B" in kwargs:
        b = kwargs.pop("B")
    else:
        raise TypeError("run() missing required argument: 'B'")

    if kwargs:
        name = next(iter(kwargs))
        raise TypeError(f"run() got an unexpected keyword argument '{name}'")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor instances")
    if a.ndim != 2 or tuple(a.shape[1:]) != (4096,):
        raise ValueError("A must have shape [M, 4096]")
    if b.ndim != 2 or tuple(b.shape) != (4096, 4096):
        raise ValueError("B must have shape [4096, 4096]")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("A and B must have dtype torch.float16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute this Triton GEMM")

    original_device = a.device
    cuda_device = a.device if a.is_cuda else (b.device if b.is_cuda else torch.device("cuda"))
    if cuda_device.type != "cuda":
        cuda_device = torch.device("cuda")

    a_gpu = a if a.device == cuda_device else a.to(cuda_device)
    b_gpu = b if b.device == cuda_device else b.to(cuda_device)
    result = _launch_gemm(a_gpu, b_gpu)

    if original_device.type != "cuda" or original_device != result.device:
        result = result.to(original_device)
    return result