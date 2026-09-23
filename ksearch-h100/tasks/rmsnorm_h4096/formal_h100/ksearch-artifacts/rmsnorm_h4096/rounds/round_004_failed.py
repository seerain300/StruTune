# solution=GPT-5.6-Sol_rmsnorm_h4096_triton_optimized_r4 score=-1.0 passed=False
I’m keeping the fused one-row reduction and narrowing the change to launch configuration. The current kernel already has strong measured performance, so the lowest-risk opportunity is selecting a lighter warp configuration for tiny batches while retaining the existing 8-warp path for medium and large workloads.import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    hidden_states,
    weight,
    output,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, 4096)
    row_offsets = row * 4096 + offsets

    x = tl.load(hidden_states + row_offsets).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_squares * 0.000244140625 + 0.00001)

    w = tl.load(weight + offsets).to(tl.float32)
    y = x * inv_rms * w
    tl.store(output + row_offsets, y)


@torch.no_grad()
def run(hidden_states, weight):
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 4096:
        raise ValueError("hidden_states must have shape [batch_size, 4096]")
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
        compute_device = hidden_states.device
        x = hidden_states.contiguous()
    else:
        x = hidden_states.contiguous().cuda()
        compute_device = x.device

    if weight.device != compute_device:
        w = weight.to(device=compute_device)
    else:
        w = weight
    w = w.contiguous()

    batch_size = x.shape[0]
    output = torch.empty_like(x)

    if batch_size > 0:
        num_warps = 4 if batch_size <= 32 else 8
        with torch.cuda.device(compute_device):
            _rmsnorm_kernel[(batch_size,)](
                x,
                w,
                output,
                num_warps=num_warps,
                num_stages=1,
            )

    if output.device != original_device:
        output = output.to(original_device)
    return output