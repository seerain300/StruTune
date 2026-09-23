import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (typically bfloat16; we will cast to float32 in-kernel)
    out_ptr,          # *output (same dtype as hidden)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
):
    # One program per row
    row_id = tl.program_id(axis=0)
    # Column indices for this row (process full row of size H)
    cols = tl.arange(0, H)

    # Load the row x and compute in float32
    x = tl.load(hidden_ptr + row_id * hidden_stride0 + cols * hidden_stride1)
    x_f32 = tl.cast(x, tl.float32)

    # Compute sum of squares and mean
    sumsq = tl.sum(x_f32 * x_f32, axis=0)
    mean = sumsq / H
    inv_rms = tl.math.rsqrt(mean + EPS)

    # Load weight vector (assumed length H) as float32
    weight = tl.load(weight_ptr + cols)
    w_f32 = tl.cast(weight, tl.float32)

    # Compute output: y = x * inv_rms * weight
    y_f32 = x_f32 * inv_rms * w_f32

    # Store output; Triton will cast to out_ptr dtype if needed
    tl.store(out_ptr + row_id * out_stride0 + cols * out_stride1, y_f32)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure contiguity
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        # Output tensor same shape and dtype as hidden
        out = torch.empty_like(hidden)

        # Grid: one program per row
        grid = (B,)

        # Epsilon as in the original PyTorch code
        EPS = 1e-5

        # Launch Triton kernel
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, EPS,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            num_warps=8, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
