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
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    # One program per row
    row_id = tl.program_id(0)

    # Base offsets for this row
    row_hidden_offset = row_id * hidden_stride0
    row_out_offset = row_id * out_stride0

    # Column indices
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # for generality; when H=4096 this is all-true

    # Load the entire row of hidden states
    x = tl.load(hidden_ptr + row_hidden_offset + cols * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store output
    tl.store(out_ptr + row_out_offset + cols * out_stride1, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguous layout
        assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA device."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == 4096, "This implementation expects hidden size of 4096."

        # Output tensor with same dtype as input hidden states
        out = torch.empty_like(hidden)

        # Grid: one program per row
        grid = (B,)

        # Launch Triton kernel (best-performing config in your environment)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=4096,
            num_warps=8,  # stabilizes performance
            num_stages=2, # stabilizes performance
        )

        return out


def run(*args):
    return ModelNew()(*args)
