import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (we'll load and cast to float32 in-kernel)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # columns per row (4096)
):
    # One program per row
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # For generality; H==BLOCK_SIZE in this task

    # Row pointers
    hidden_row = hidden_ptr + row_id * hidden_stride0
    out_row = out_ptr + row_id * out_stride0

    # Load the entire row and cast to float32 for math
    x = tl.load(hidden_row + cols * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight as float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store (cast handled by Triton store to out_ptr element type)
    tl.store(out_row + cols * out_stride1, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        # Ensure contiguous for simple striding
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == 4096, "This implementation expects hidden size 4096."

        # Output tensor in the same dtype as hidden (original module returns same dtype)
        out = torch.empty_like(hidden)

        # Strides (in elements)
        hidden_stride0 = hidden.stride(0)
        hidden_stride1 = hidden.stride(1)
        out_stride0 = out.stride(0)
        out_stride1 = out.stride(1)

        # Kernel launch: one program per row
        grid = (B,)
        EPS = 1e-5

        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, EPS,
            hidden_stride0, hidden_stride1,
            out_stride0, out_stride1,
            BLOCK_SIZE=4096,
            num_warps=8,  # best observed in this environment
            num_stages=2, # best observed in this environment
        )
        return out


def run(*args):
    return ModelNew()(*args)
