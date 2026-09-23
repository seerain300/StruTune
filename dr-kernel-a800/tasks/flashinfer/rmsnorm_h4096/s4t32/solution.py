import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16), shape [B, H]
    weight_ptr,       # *const weight tensor (e.g., bfloat16), shape [H]
    out_ptr,          # *output tensor, shape [B, H], same dtype as hidden
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (expected 4096 in this task)
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
    sumsq = tl.sum(x_f32 * x_f32, axis=0)  # scalar
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # scalar

    # Load weight vector (length H), cast to float32
    weight = tl.load(weight_ptr + cols)
    weight_f32 = tl.cast(weight, tl.float32)

    # Compute output: y = x * inv_rms * weight
    y_f32 = x_f32 * inv_rms * weight_f32

    # Store to output; Triton will cast y_f32 to out_ptr's element dtype as needed
    tl.store(out_ptr + row_id * out_stride0 + cols * out_stride1, y_f32)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        # Output tensor with same dtype and shape as hidden_states
        out = torch.empty_like(hidden_states)

        # Launch parameters that previously performed best
        num_warps = 8
        num_stages = 2

        # Launch Triton kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden_states,
            weight,
            out,
            B, H,
            1e-5,  # EPS
            hidden_states.stride(0), hidden_states.stride(1),
            out.stride(0), out.stride(1),
            num_warps=num_warps, num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
