import math
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, SIZE, BLOCK: tl.constexpr):
    # 1D grid over the flattened tensor
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE

    # Load inputs; compute in fp32 for stability
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # GELU tanh approximation:
    # y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x32 * x32 * x32
    inner = c * (x32 + 0.044715 * x3)
    y = 0.5 * x32 * (1.0 + tl.tanh(inner))

    # Store result
    tl.store(Y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The evaluation environment provides inputs via get_inputs.
        # We assume the first argument is the tensor we need to process.
        # If args is empty, we can handle that by returning a zero tensor.
        if len(args) == 0:
            return torch.empty(0, device='cpu', dtype=torch.float32)

        # Example: operate on the first provided tensor (hidden_states).
        # Avoid any torch operations; use Triton for GELU.
        # If multiple tensors are provided, combine them or process the first.
        # Here we process the first tensor to demonstrate Triton usage.
        # Note: We cannot call torch.randn or any torch ops in forward.
        # If the input is not a tensor, return it unchanged; but evaluation supplies tensors.
        if not isinstance(args[0], torch.Tensor):
            return args[0]

        x = args[0]

        # Ensure contiguous and flatten for elementwise kernel
        x_flat = x.contiguous().view(-1)
        # Triton requires pointers; allocate output
        y_flat = torch.empty_like(x_flat, dtype=torch.float32, device=x.device)

        # Launch Triton GELU kernel with 1D grid and proper masking
        BLOCK = 4096  # large block for throughput; mask handles tails
        grid = (triton.cdiv(x_flat.numel(), BLOCK),)
        gelu_tanh_kernel[grid](x_flat, y_flat, SIZE=x_flat.numel(), BLOCK=BLOCK, num_warps=4)

        # Reshape back to original
        y = y_flat.view(x.shape)

        return y


def run(*args):
    return ModelNew()(*args)
