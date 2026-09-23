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
    # Optional safety in case grid is larger than batch_size (not expected here)
    if pid >= batch_size:
        return

    row_offset = pid * hidden_size

    # Vector of column offsets (0..BLOCK_SIZE-1). With hidden_size=4096, mask is all true.
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Pass 1: sum of squares
    # Load x as its original dtype, cast to float32 for accumulation.
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)
    sumsq = tl.sum(x32 * x32, axis=0)

    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and store
    w = tl.load(weight_ptr + offs, mask=mask, other=0).to(tl.float32)
    x32 = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0).to(tl.float32)
    y32 = x32 * inv_rms * w
    # Store back; Triton will cast to the pointer's dtype if needed.
    tl.store(out_ptr + row_offset + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and expected shapes
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
        assert hidden_states.ndim == 2, "hidden_states must be [batch_size, hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Prepare output
        out = torch.empty_like(hidden_states)

        # Compute in float32 inside the kernel for numerical stability; original code does the same.
        # Use a robust launch configuration for memory-bound patterns.
        BLOCK_SIZE = 4096  # vectorize across the entire row
        EPS = 1e-5

        grid = (batch_size,)  # one program per row

        # Launch Triton kernel
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out,
            batch_size, hidden_size, EPS,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,   # higher occupancy for better throughput
            num_stages=1,   # simple, memory-bound kernel
        )

        return out


def run(*args):
    return ModelNew()(*args)
