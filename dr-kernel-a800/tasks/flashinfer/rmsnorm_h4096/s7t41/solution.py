import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms.
# - Pass 2: scale and store output, multiplying by weight.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             batch_size, hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute sum of squares in float32, then inv_rms
    x = tl.load(hidden_ptr + pid * hidden_size + offs, mask=offs < hidden_size, other=0.0)
    # x is loaded as the original dtype (e.g., bfloat16), convert to fp32 for math
    x_fp32 = x.to(tl.float32)
    sumsq = tl.sum(x_fp32 * x_fp32)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and store
    w = tl.load(weight_ptr + offs, mask=offs < hidden_size, other=0.0).to(tl.float32)
    y = (x_fp32 * inv_rms) * w
    # Store results (fp32), final casting to output dtype is done on host after kernel
    tl.store(out_ptr + pid * hidden_size + offs, y, mask=offs < hidden_size)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        # We require the hidden_size to be 4096 as in the original code
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Allocate fp32 output buffer for numeric stability and simplicity
        out_fp32 = torch.empty_like(hidden, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        # Use BLOCK_SIZE = 4096 (constexpr), with mask to be safe (even though always true here)
        _row_fused_scale_kernel[grid](
            hidden, w, out_fp32,
            batch_size, hidden_size, 1e-5,  # EPS as float, hidden_size constexpr
            BLOCK_SIZE=4096,
            num_warps=16,  # good for memory-bound ops
            num_stages=1
        )

        # Cast back to the original dtype to match the original behavior
        out = out_fp32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
