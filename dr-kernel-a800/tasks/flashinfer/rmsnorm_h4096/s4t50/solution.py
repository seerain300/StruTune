import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (we'll load and cast to float32 in-kernel)
    out_ptr,          # *output (float32)
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
    row_hidden_offset = row_id * hidden_stride0
    row_out_offset = row_id * out_stride0

    cols = tl.arange(0, BLOCK_SIZE)  # 0..4095
    mask = cols < H  # true for H=4096; kept for generality

    # Load hidden row and weight; cast to float32 for math
    x = tl.load(hidden_ptr + row_hidden_offset + cols * hidden_stride1, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    x_f32 = tl.cast(x, tl.float32)
    w_f32 = tl.cast(w, tl.float32)

    # Compute sum of squares and mean for this row
    x2 = x_f32 * x_f32
    sumsq = tl.sum(x2, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # scalar per row

    # Scale and store (output in float32)
    y = x_f32 * inv_rms * w_f32
    tl.store(out_ptr + row_out_offset + cols * out_stride1, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure contiguous for simple stride handling
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Expect hidden_size == 4096 as per original assertion
        B, H = hidden.shape
        assert H == 4096, "This kernel expects hidden size 4096"

        # Allocate output in float32 for storage
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Grid: one program per row
        grid = (B,)

        # Launch Triton kernel with tuned parameters (best observed in your environment)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out_f32,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out_f32.stride(0), out_f32.stride(1),
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=2,
        )

        # Cast back to original dtype to match the original PyTorch behavior
        return out_f32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
