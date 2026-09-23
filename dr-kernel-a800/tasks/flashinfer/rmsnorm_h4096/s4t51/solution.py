import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (cast to float32 in-kernel)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    # One program per row
    row_id = tl.program_id(0)

    # Column offsets for this row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # generality; for H=4096, mask is all-true but kept for safety

    # Compute pointers for this row
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0 + cols * hidden_stride1
    out_row_ptr = out_ptr + row_id * out_stride0 + cols * out_stride1

    # Load hidden row and cast to float32 for math
    x = tl.load(hidden_row_ptr, mask=mask, other=0.0)
    x_f32 = tl.cast(x, tl.float32)

    # Reduce to compute sum of squares
    sumsq = tl.sum(x_f32 * x_f32, axis=0)  # scalar
    mean = sumsq / H
    inv_rms = tl.math.rsqrt(mean + EPS)

    # Load weight and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w_f32 = tl.cast(w, tl.float32)

    # Compute output
    y = x_f32 * inv_rms * w_f32

    # Store output (Triton will cast to out_ptr element type if needed)
    tl.store(out_row_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure contiguous for predictable strides
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Allocate output (same shape and dtype as hidden)
        out = torch.empty_like(hidden)

        # Shapes and strides
        B, H = hidden.shape
        assert H == 4096, "This optimized kernel expects hidden size 4096."

        # Grid: one program per row
        grid = (B,)

        # Launch Triton kernel with the best-performing configuration
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
