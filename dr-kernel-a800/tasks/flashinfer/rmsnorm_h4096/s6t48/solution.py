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
    BLOCK_SIZE: tl.constexpr  # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row
    # Compute sum of squares across the row to get inv_rms
    sumsq = 0.0
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        # Load x (float32) for this row tile
        x = tl.load(hidden_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + eps)  # scalar per row

    # Second pass: scale and write output
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row_id * hidden_size + cols, y, mask=mask)


def _dtype_code(dtype: torch.dtype) -> int:
    # 0: fp32, 1: fp16, 2: bf16
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2
    else:
        # default to fp32 if some other type
        return 0


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure shapes: hidden_states [batch, 4096], weight [4096]
        assert hidden_states.dim() == 2 and hidden_states.shape[1] == 4096, "hidden_states must be [batch, 4096]"
        assert weight.dim() == 1 and weight.shape[0] == 4096, "weight must be [4096]"

        # Cast to float32 for computation (matches original run behavior)
        hidden_f32 = hidden_states.to(torch.float32)
        weight_f32 = weight.to(torch.float32)

        batch_size = hidden_f32.shape[0]
        hidden_size = hidden_f32.shape[1]
        eps = 1e-5

        # Allocate output in the same dtype as input
        out = torch.empty_like(hidden_states)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _rms_scale_kernel[grid](
            hidden_f32, weight_f32, out,
            batch_size, hidden_size, eps,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=3
        )

        return out


def run(*args):
    return ModelNew()(*args)
