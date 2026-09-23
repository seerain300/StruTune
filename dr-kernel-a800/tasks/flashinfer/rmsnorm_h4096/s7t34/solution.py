import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms.
# - Pass 2: scale and store, multiply by weight.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr, inv_rms_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * hidden_size

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # always true when BLOCK_SIZE == hidden_size; kept for generality

    # Pass 1: compute sum of squares in float32
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    x2 = x32 * x32
    sum_x2 = tl.sum(x2, axis=0)
    mean_x2 = sum_x2 / hidden_size
    inv_rms = tl.math.rsqrt(mean_x2 + EPS)
    # Store inv_rms as a single scalar per row
    tl.store(inv_rms_ptr + pid, inv_rms)

    # Pass 2: scale and store
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w32 = w.to(tl.float32)
    inv = tl.load(inv_rms_ptr + pid)  # scalar load
    y32 = x32 * inv * w32
    y = y32.to(x.dtype)  # cast back to original dtype for output
    tl.store(out_ptr + row_offset + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
        assert hidden_states.dim() == 2, "hidden_states must be [batch_size, hidden_size]"
        assert weight.dim() == 1 and weight.numel() == hidden_states.shape[1], "weight must be [hidden_size]"

        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "This kernel assumes hidden_size == 4096"

        # Output and per-row inv_rms buffer
        out = torch.empty_like(hidden_states)
        inv_rms = torch.empty((batch_size,), device=hidden_states.device, dtype=torch.float32)

        # Heuristic launch configuration
        if batch_size >= 64:
            num_warps = 16
        else:
            num_warps = 8
        num_stages = 2  # modest pipelining

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out, inv_rms,
            hidden_size=hidden_size, EPS=1e-5,
            BLOCK_SIZE=hidden_size,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return out


def run(*args):
    return ModelNew()(*args)
