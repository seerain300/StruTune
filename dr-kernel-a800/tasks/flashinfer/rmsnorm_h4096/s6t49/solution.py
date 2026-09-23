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
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row
    base = row_id * hidden_size

    # First pass: compute sum of squares across the row
    sumsq = 0.0
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + base + cols, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + eps)  # 1 / sqrt(mean + eps)

    # Second pass: scale and write output
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + base + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w

        # Cast to target dtype and store
        if out_dtype_code == 0:
            out_val = y  # fp32
        elif out_dtype_code == 1:
            out_val = y.to(tl.float16)
        elif out_dtype_code == 2:
            out_val = y.to(tl.bfloat16)
        else:
            out_val = y  # default fp32
        tl.store(out_ptr + base + cols, out_val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Input checks
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch_size, hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "This implementation expects hidden_size == 4096"
        assert weight.shape == (hidden_size,), "weight must be 1D of length hidden_size"

        # Prepare inputs: compute in float32 for numerical stability
        hidden_in = hidden_states.contiguous().to(torch.float32)
        weight_in = weight.contiguous().to(torch.float32)

        # Output tensor with same dtype and shape as hidden_states
        out = torch.empty_like(hidden_states)

        # Grid: one program per row
        grid = (batch_size,)

        # Execution parameters tuned for hidden_size=4096
        BLOCK_SIZE = 1024
        num_warps = 8
        num_stages = 3

        # Determine output dtype code
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            out_dtype_code = 0  # default to fp32

        # Launch Triton kernel
        _rms_scale_kernel[grid](
            hidden_in, weight_in, out,
            batch_size, hidden_size, 1e-5, out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return out


def run(*args):
    return ModelNew()(*args)
