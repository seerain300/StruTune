import math
import torch
import triton
import triton.language as tl


@triton.jit
def cosine_kernel(inp_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Compute elementwise cosine of input and write to output.
    inp_ptr: *const bfloat16
    out_ptr: *mut bfloat16
    N: total number of elements
    BLOCK: constexpr chunk size for parallelization
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load input (assume bf16)
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    # Compute cosine
    y = tl.cos(x)
    # Store output
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure we have at least one tensor argument; the original get_inputs passes hidden_states first.
        if len(args) == 0:
            raise RuntimeError("forward requires at least one input tensor")
        # We will use the first argument as hidden_states (expected shape [num_tokens, hidden_size], dtype bfloat16)
        hidden_states = args[0]
        # Validate device: Triton requires CUDA tensors
        if hidden_states.device.type != "cuda":
            raise RuntimeError("ModelNew requires CUDA device for Triton kernel")

        # Prepare output
        out = torch.empty_like(hidden_states)

        # Flatten for 1D kernel; Triton will write back with the same shape
        N = hidden_states.numel()
        BLOCK = 1024  # chunk size; tuneable
        grid = (triton.cdiv(N, BLOCK),)

        # Launch the Triton kernel
        cosine_kernel[grid](
            hidden_states,  # inp_ptr
            out,             # out_ptr
            N,               # total elements
            BLOCK=BLOCK,
        )

        # Return the computed output
        return out


def run(*args):
    return ModelNew()(*args)
