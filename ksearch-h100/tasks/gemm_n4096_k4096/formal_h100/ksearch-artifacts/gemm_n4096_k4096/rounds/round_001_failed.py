# solution=GPT-5.6-Sol_gemm_n4096_k4096_triton_optimized_r1 score=-1.0 passed=False
I’m implementing the requested latency-family path with a dedicated skinny-M kernel that gives each row persistent ownership across the fixed N tiles, while retaining a conventional tiled fallback for larger M. The wrapper will normalize device placement and restore outputs to the input device.import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    for tile_idx in range(16):
        cols = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < 4096
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_idx in range(64):
            k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k < 4096

            a = tl.load(
                a_ptr + rows[:, None] * 4096 + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptr + cols[None, :] * 4096 + k[:, None],
                mask=col_mask[None, :] & k_mask[:, None],
                other=0.0,
            )

            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

        tl.store(
            c_ptr + rows[:, None] * 4096 + cols[None, :],
            acc,
            mask=row_mask[:, None] & col_mask[None, :],
        )


@triton.jit
def _tiled_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < 4096

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(64):
        k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k < 4096

        a = tl.load(
            a_ptr + rows[:, None] * 4096 + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + cols[None, :] * 4096 + k[:, None],
            mask=col_mask[None, :] & k_mask[:, None],
            other=0.0,
        )

        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    tl.store(
        c_ptr + rows[:, None] * 4096 + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def run(*args, **kwargs):
    if len(args) > 2:
        raise TypeError("run() takes at most 2 positional arguments")

    A = args[0] if len(args) > 0 else kwargs.pop("A", None)
    B = args[1] if len(args) > 1 else kwargs.pop("B", None)

    if A is None:
        A = kwargs.pop("a", None)
    if B is None:
        B = kwargs.pop("b", None)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"run() got an unexpected keyword argument '{unexpected}'")
    if A is None or B is None:
        raise TypeError("run() requires A and B")

    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor objects")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must both be 2D tensors")
    if A.shape[1] != 4096:
        raise ValueError("A must have shape [M, 4096]")
    if B.shape != (4096, 4096):
        raise ValueError("B must have shape [4096, 4096]")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError("A and B must have dtype torch.float16")

    original_device = A.device
    if not torch.cuda.is_available():
        if A.is_cuda or B.is_cuda:
            raise RuntimeError("CUDA is unavailable but a GPU tensor was provided")
        raise RuntimeError("CUDA is required to run this Triton kernel")

    A_gpu = A.cuda().contiguous()
    B_gpu = B.cuda().contiguous()
    M = A_gpu.shape[0]
    C_gpu = torch.empty((M, 4096), device=A_gpu.device, dtype=torch.float16)

    if M <= 32:
        grid = (triton.cdiv(M, 8),)
        _skinny_gemm_kernel[grid](
            A_gpu,
            B_gpu,
            C_gpu,
            M,
            BLOCK_M=8,
            BLOCK_N=256,
            BLOCK_K=64,
            num_warps=4,
            num_stages=2,
        )
    else:
        grid = (triton.cdiv(M, 16), triton.cdiv(4096, 128))
        _tiled_gemm_kernel[grid](
            A_gpu,
            B_gpu,
            C_gpu,
            M,
            BLOCK_M=16,
            BLOCK_N=128,
            BLOCK_K=64,
            num_warps=8,
            num_stages=3,
        )

    return C_gpu.to(original_device)