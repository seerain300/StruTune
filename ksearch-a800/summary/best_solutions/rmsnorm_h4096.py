# task: rmsnorm_h4096
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=14/14 geomean=3.471x
# feedback best (5-workload sample during search): 4.853x
# torch fallback audit: 干净 (-)
# tokens: 952,384

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    hidden_states,
    weight,
    output,
    HIDDEN_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, HIDDEN_SIZE)
    offsets = row * HIDDEN_SIZE + columns

    x = tl.load(hidden_states + offsets).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_squares * (1.0 / HIDDEN_SIZE) + EPS)

    w = tl.load(weight + columns).to(tl.float32)
    tl.store(output + offsets, x * inv_rms * w)


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
    if hidden_states.device.type not in ("cpu", "cuda"):
        raise ValueError("hidden_states must be on a CPU or CUDA device")
    if weight.device.type not in ("cpu", "cuda"):
        raise ValueError("weight must be on a CPU or CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the Triton RMSNorm kernel")

    original_device = hidden_states.device

    if hidden_states.is_cuda:
        target_device = hidden_states.device
    elif weight.is_cuda:
        target_device = weight.device
    else:
        target_device = torch.device("cuda", torch.cuda.current_device())

    if hidden_states.device != target_device:
        hidden_states_gpu = hidden_states.cuda(device=target_device)
    else:
        hidden_states_gpu = hidden_states

    if weight.device != target_device:
        weight_gpu = weight.cuda(device=target_device)
    else:
        weight_gpu = weight

    hidden_states_gpu = hidden_states_gpu.contiguous()
    weight_gpu = weight_gpu.contiguous()
    output_gpu = torch.empty_like(hidden_states_gpu)

    batch_size = hidden_states_gpu.shape[0]
    if batch_size != 0:
        with torch.cuda.device(target_device):
            _rmsnorm_kernel[(batch_size,)](
                hidden_states_gpu,
                weight_gpu,
                output_gpu,
                HIDDEN_SIZE=4096,
                EPS=1e-5,
                num_warps=8,
                num_stages=1,
            )

    if output_gpu.device != original_device:
        return output_gpu.to(original_device)
    return output_gpu