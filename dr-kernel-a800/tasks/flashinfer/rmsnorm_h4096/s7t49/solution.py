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
    row_offset = pid * hidden_size

    # Vector of column offsets
    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute sum of squares across the row
    sum_sq = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        idx = col + offs
        # With BLOCK_SIZE == hidden_size, this loop runs once; mask not needed in practice.
        x = tl.load(hidden_ptr + row_offset + idx)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)  # scalar per row

    # Pass 2: scale and store
    for col in range(0, hidden_size, BLOCK_SIZE):
        idx = col + offs
        x = tl.load(hidden_ptr + row_offset + idx).to(tl.float32)
        w = tl.load(weight_ptr + idx).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + row_offset + idx, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Convert inputs to float32 for computation
        hidden_f32 = hidden_states.to(torch.float32)
        weight_f32 = weight.to(torch.float32)

        # Output buffer in float32
        out_f32 = torch.empty((batch_size, hidden_size), device=hidden_states.device, dtype=torch.float32)

        # Launch: one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_f32, weight_f32, out_f32,
            batch_size, hidden_size, 1e-5, 4096,
            num_warps=32, num_stages=1
        )

        # Cast back to original dtype to match PyTorch behavior
        return out_f32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
