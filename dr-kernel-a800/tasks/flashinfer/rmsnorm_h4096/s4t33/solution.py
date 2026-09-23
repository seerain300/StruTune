import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16), shape [B, H]
    weight_ptr,       # *const weight tensor (e.g., bfloat16/bfloat32), shape [H]
    out_ptr,          # *output tensor, shape [B, H], same dtype as hidden
    B: tl.int32,      # batch size (not strictly needed in kernel, but kept for clarity)
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden (elements)
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden (elements)
    out_stride0: tl.int32,     # stride along batch dim for output (elements)
    out_stride1: tl.int32,     # stride along hidden dim for output (elements)
):
    # One program per row
    row_id = tl.program_id(axis=0)
    cols = tl.arange(0, H)

    # Load the row x (vector of size H), cast to float32 for math
    x = tl.load(hidden_ptr + row_id * hidden_stride0 + cols * hidden_stride1)
    x_f32 = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x_f32 * x_f32, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols)
    w_f32 = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * weight
    y = x_f32 * inv_rms * w_f32

    # Store result to output (automatic cast to out_ptr dtype if needed)
    tl.store(out_ptr + row_id * out_stride0 + cols * out_stride1, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA for Triton kernel."

        # Ensure contiguous for efficient memory access
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = hidden.shape

        # Output tensor with same dtype and shape as hidden
        out = torch.empty_like(hidden)

        # Epsilon as float32
        EPS = 1e-5

        # Launch Triton kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, w, out,
            B, H, EPS,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            num_warps=8,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
