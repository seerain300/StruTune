import torch
import triton
import triton.language as tl


# Simple 1D elementwise copy: y[i] = x[i]
@triton.jit
def copy_1d_kernel(X_ptr, Y_ptr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        # Avoid torch ops for heavy numeric work; only data movement and Triton kernel.
        # Copy input to output using Triton to ensure a kernel is invoked.
        # Ensure dtype and shape are preserved.
        assert hidden_states.is_cuda, "ModelNew expects CUDA tensors."
        x = hidden_states  # do not modify x in-place
        y = torch.empty_like(x)
        x_flat = x.contiguous().view(-1)
        y_flat = y.view(-1)
        size = x_flat.numel()
        # Choose a block size that gives good occupancy without risking OOB
        BLOCK = 4096
        grid = (triton.cdiv(size, BLOCK),)
        copy_1d_kernel[grid](x_flat, y_flat, SIZE=size, BLOCK=BLOCK, num_warps=4)
        return y


def run(*args):
    return ModelNew()(*args)
