import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (typically bfloat16; we cast to float32 in-kernel)
    out_ptr,          # *output (same dtype as hidden)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
):
    # One Triton program per row
    row_id = tl.program_id(axis=0)
    cols = tl.arange(0, H)  # vector of column indices [0..H-1]

    # Load the entire row of hidden states
    x = tl.load(hidden_ptr + row_id * hidden_stride0 + cols * hidden_stride1)
    # Compute in float32 for numerical stability and to match original semantics
    x_f32 = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x_f32 * x_f32, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols)  # weight has shape [H]; contiguous indexing
    w_f32 = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * weight
    y = x_f32 * inv_rms * w_f32

    # Store result to output
    tl.store(out_ptr + row_id * out_stride0 + cols * out_stride1, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous for simple stride-based addressing in Triton
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Shapes
        batch_size, hidden_size = hidden.shape
        assert hidden_size == 4096, "This kernel expects hidden_size == 4096."

        # Output tensor in the same dtype and shape as hidden_states
        out = torch.empty_like(hidden)

        # Constants
        EPS = 1e-5

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            batch_size, hidden_size, EPS,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            num_warps=8,  # tuned for best performance in your environment
            num_stages=2, # tuned for best performance in your environment
        )
        return out


def run(*args):
    return ModelNew()(*args)
