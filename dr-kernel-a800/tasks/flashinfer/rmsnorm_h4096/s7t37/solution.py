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
    # Optional safety: ensure pid is within range (in case grid is over-provisioned)
    if pid >= batch_size:
        return

    row_offset = pid * hidden_size

    # Vectorized offsets across the row
    offs = tl.arange(0, BLOCK_SIZE)
    # With hidden_size == 4096, offs < hidden_size is always true; kept for generality
    mask = offs < hidden_size

    # Pass 1: compute sum of squares in float32
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    sq = x32 * x32
    sum_sq = tl.sum(sq, axis=0)
    mean_sq = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Pass 2: scale and store
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    y32 = x32 * inv_rms * w
    # Store back in the original dtype of hidden_ptr (out_ptr has same dtype as hidden_ptr)
    tl.store(out_ptr + row_offset + offs, y32.to(x.dtype), mask=mask)

# Host-side wrapper that launches the Triton kernel
def triton_run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Expect 2D hidden_states [batch_size, 4096], and 1D weight [4096]
    assert hidden_states.ndim == 2 and hidden_states.shape[1] == 4096
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    # Ensure contiguous tensors
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    # Allocate output tensor with same dtype and shape as input
    out = torch.empty_like(hidden)

    # Grid: one program per row
    grid = (batch_size,)

    # Launch kernel. Using BLOCK_SIZE=4096 and tuning for memory throughput.
    _row_fused_scale_kernel[grid](
        hidden, weight, out,
        batch_size, hidden_size, 1e-5,
        BLOCK_SIZE=4096,
        num_warps=32,   # higher warps for better occupancy on memory-bound workloads
        num_stages=1    # simple pipeline; reduces register pressure
    )
    return out

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two inputs: hidden_states [batch, 4096] and weight [4096]
        # The original Model.forward signature is run(hidden_states, weight).
        if len(args) != 2:
            raise ValueError("ModelNew expects two inputs: hidden_states and weight.")
        hidden_states, weight = args
        return triton_run(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
