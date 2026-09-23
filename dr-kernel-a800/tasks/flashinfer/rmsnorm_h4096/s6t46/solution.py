import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. We read x twice: once to compute sum of squares, once to write output.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row
    # Offsets for this row
    cols = tl.arange(0, BLOCK_SIZE)
    row_start = row_id * hidden_size

    # First pass: accumulate sum of squares
    sumsq = 0.0
    # Loop over the row in chunks of BLOCK_SIZE
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        offs = col_start + cols
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_start + offs, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: scale and write output
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        offs = col_start + cols
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_start + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        # Store (implicit cast to out_ptr dtype)
        tl.store(out_ptr + row_id * hidden_size + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
        # Shapes and asserts
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"
        # Ensure contiguous tensors
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Compute in float32
        hidden_f32 = hidden.to(torch.float32)
        weight_f32 = weight.to(torch.float32)

        # Output tensor with same dtype as input hidden_states
        out = torch.empty((batch_size, hidden_size), device=hidden.device, dtype=hidden.dtype)

        # Grid: one program per row
        grid = (batch_size,)

        # Launch Triton kernel
        _rms_scale_kernel[grid](
            hidden_f32, weight_f32, out,
            batch_size, hidden_size, 1e-5,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=3,
        )
        return out


def run(*args):
    return ModelNew()(*args)
