# solution=GPT-5.6-Sol_rmsnorm_h4096_triton_optimized_r12 score=-1.0 passed=False
I’m going to preserve the established fused row-per-program kernel and target the remaining throughput regime through launch configuration. The persistent performance signal points to reducing per-row warp pressure for large batches, while retaining the latency-oriented small-batch choices and the existing validation/device behavior.import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    hidden_states,
    weight,
    output,
    x_row_stride: tl.constexpr,
    x_col_stride: tl.constexpr,
    weight_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    out_col_stride: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, HIDDEN_SIZE)

    x = tl.load(
        hidden_states + row * x_row_stride + cols * x_col_stride
    ).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    inv_rms = tl.rsqrt(variance + EPS)

    w = tl.load(weight + cols * weight_stride).to(tl.float32)
    y = x * inv_rms * w

    tl.store(
        output + row * out_row_stride + cols * out_col_stride,
        y,
    )


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
        raise RuntimeError("CUDA is required to execute the Triton RMSNorm kernel")

    original_device = hidden_states.device

    if hidden_states.is_cuda:
        execution_device = hidden_states.device
    elif weight.is_cuda:
        execution_device = weight.device
    else:
        execution_device = torch.device("cuda")

    hidden_states_gpu = hidden_states.to(execution_device)
    weight_gpu = weight.to(execution_device)
    output_gpu = torch.empty_like(hidden_states_gpu)

    batch_size = hidden_states_gpu.shape[0]
    if batch_size == 0:
        return output_gpu.to(original_device)

    if batch_size <= 32:
        num_warps = 4
    elif batch_size <= 512:
        num_warps = 8
    else:
        num_warps = 4

    with torch.cuda.device(execution_device):
        _rmsnorm_kernel[(batch_size,)](
            hidden_states_gpu,
            weight_gpu,
            output_gpu,
            hidden_states_gpu.stride(0),
            hidden_states_gpu.stride(1),
            weight_gpu.stride(0),
            output_gpu.stride(0),
            output_gpu.stride(1),
            HIDDEN_SIZE=4096,
            EPS=1e-5,
            num_warps=num_warps,
        )

    return output_gpu.to(original_device)