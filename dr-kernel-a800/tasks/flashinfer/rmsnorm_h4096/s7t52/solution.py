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

    # Pass 1: compute sum of squares in float32 and derive inv_rms
    sumsq = tl.zeros((), dtype=tl.float32)
    x = tl.load(hidden_ptr + pid * hidden_size + offs, mask=offs < hidden_size, other=0.0)
    x = x.to(tl.float32)
    sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and store output: y = x * inv_rms * weight
    w = tl.load(weight_ptr + offs, mask=offs < hidden_size, other=0.0)
    w = w.to(tl.float32)
    y = x * inv_rms * w
    tl.store(out_ptr + pid * hidden_size + offs, y, mask=offs < hidden_size)


def _run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure CUDA tensors and contiguity
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    hidden = hidden_states.contiguous()
    w = weight.contiguous()

    batch_size, hidden_size = hidden.shape
    assert hidden_size == 4096, "hidden_size must be 4096."

    # Output in float32 (we'll cast to original dtype after kernel)
    out_fp32 = torch.empty_like(hidden, dtype=torch.float32)

    # Launch: one program per row
    grid = (batch_size,)
    # Use the configuration that previously showed best performance
    _row_fused_scale_kernel[grid](
        hidden, w, out_fp32,
        batch_size,
        hidden_size=hidden_size,
        EPS=1e-5,
        BLOCK_SIZE=hidden_size,
        num_warps=16,
        num_stages=1,
    )

    # Cast back to original dtype to match the original behavior
    return out_fp32.to(hidden_states.dtype)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two tensors: hidden_states and weight
        if len(args) != 2:
            raise RuntimeError("ModelNew.forward expects two tensors: hidden_states and weight.")
        hidden_states, weight = args
        return _run(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
