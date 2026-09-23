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

    # Vector of column offsets (0..BLOCK_SIZE-1)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # kept for generality; for hidden_size == BLOCK_SIZE, always true

    # Pass 1: compute sum of squares across the row
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0)
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    sum_sq = tl.sum(x2, axis=0)

    # Compute inv_rms for this row
    mean_sq = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Pass 2: scale and store
    y = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0)
    w = tl.load(weight_ptr + offs, mask=mask, other=0).to(tl.float32)
    y_f32 = y.to(tl.float32) * inv_rms
    y_f32 = y_f32 * w
    # Store; out_ptr dtype determines final output dtype (host ensures it matches desired cast)
    tl.store(out_ptr + row_offset + offs, y_f32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure contiguity and expected shapes
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch, hidden_size]"
        assert weight.dim() == 1, "weight must be 1D [hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "This optimized kernel expects hidden_size == 4096"

        # Make sure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        # Output tensor with the same dtype as input hidden_states
        out = torch.empty_like(hidden_states)

        # Launch Triton kernel: one program per row
        # Choose robust launch params for memory-bound kernels
        BLOCK_SIZE = hidden_size  # 4096
        EPS = 1e-5

        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out,
            batch_size, hidden_size, EPS,
            BLOCK_SIZE,
            num_warps=16,  # proven robust choice
            num_stages=1   # good for simple, memory-bound kernels
        )
        return out


def run(*args):
    return ModelNew()(*args)
