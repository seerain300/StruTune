import torch
import triton
import triton.language as tl


@triton.jit
def gelu_tanh_kernel(x_ptr, y_ptr, SIZE, BLOCK: tl.constexpr):
    # 1D grid over flattened elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU (tanh approximation):
    # y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure forward uses Triton only. Assume at least one tensor input (hidden_states).
        if len(args) == 0:
            return None

        x = args[0]

        # Ensure CUDA and contiguous; Triton does not support CPU tensors in kernels
        if x.device.type != 'cuda':
            x = x.to('cuda')

        x = x.contiguous().to(torch.float32)
        SIZE = x.numel()
        y = torch.empty_like(x, dtype=torch.float32, device=x.device)

        BLOCK = 1024
        grid = (triton.cdiv(SIZE, BLOCK),)
        gelu_tanh_kernel[grid](x, y, SIZE, BLOCK=BLOCK, num_warps=4)

        return y


def run(*args):
    return ModelNew()(*args)
