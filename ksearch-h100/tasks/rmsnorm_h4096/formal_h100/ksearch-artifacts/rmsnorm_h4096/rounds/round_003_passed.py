# solution=GPT-5.6-Sol_rmsnorm_h4096_triton_optimized_r3 score=4.409379701017741 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    hidden_states,
    weight,
    output,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVICT_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = row * 4096 + offsets

    x = tl.load(
        hidden_states + row_offsets,
        eviction_policy="evict_first",
    ).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_squares * 0.000244140625 + EPS)

    if EVICT_WEIGHT:
        w = tl.load(
            weight + offsets,
            eviction_policy="evict_last",
        )
    else:
        w = tl.load(weight + offsets)

    tl.store(output + row_offsets, x * inv_rms * w)


@torch.no_grad()
def run(hidden_states, weight):
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [batch_size, 4096]")
    if hidden_states.shape[1] != 4096:
        raise ValueError("hidden_states hidden size must be 4096")
    if weight.ndim != 1 or weight.shape[0] != 4096:
        raise ValueError("weight must have shape [4096]")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must have dtype torch.bfloat16")
    if weight.dtype != torch.bfloat16:
        raise TypeError("weight must have dtype torch.bfloat16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute rmsnorm_h4096")

    original_device = hidden_states.device

    if hidden_states.is_cuda:
        execution_device = hidden_states.device
        hidden_gpu = hidden_states
    elif weight.is_cuda:
        execution_device = weight.device
        hidden_gpu = hidden_states.cuda(device=execution_device)
    else:
        execution_device = torch.device("cuda", torch.cuda.current_device())
        hidden_gpu = hidden_states.cuda(device=execution_device)

    weight_gpu = (
        weight
        if weight.device == execution_device
        else weight.cuda(device=execution_device)
    )

    hidden_gpu = hidden_gpu.contiguous()
    weight_gpu = weight_gpu.contiguous()
    output_gpu = torch.empty_like(hidden_gpu)

    batch_size = hidden_gpu.shape[0]
    if batch_size == 0:
        return output_gpu.to(original_device)

    if batch_size <= 8:
        num_warps = 8
    elif batch_size <= 32:
        num_warps = 4
    else:
        num_warps = 8

    _rmsnorm_kernel[(batch_size,)](
        hidden_gpu,
        weight_gpu,
        output_gpu,
        EPS=1e-5,
        BLOCK_SIZE=4096,
        EVICT_WEIGHT=batch_size > 32,
        num_warps=num_warps,
        num_stages=1,
    )

    if output_gpu.device != original_device:
        return output_gpu.to(original_device)
    return output_gpu