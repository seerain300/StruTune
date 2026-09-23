import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (we'll load and cast to float32 in-kernel)
    out_ptr,          # *output (same shape as hidden; dtype determined by out tensor)
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
    if row_id >= B:
        return

    # Column indices for this row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # robust even if H < BLOCK_SIZE (not used here, but kept for safety)

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0
    out_row_ptr = out_ptr + row_id * out_stride0

    # Load entire row (as native dtype), cast to float32 for math
    x = tl.load(hidden_row_ptr + cols * hidden_stride1, mask=mask, other=0.0)
    x_f32 = tl.cast(x, tl.float32)

    # Compute sum of squares for this row
    sumsq = tl.sum(x_f32 * x_f32, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w_f32 = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * weight
    y_f32 = x_f32 * inv_rms * w_f32

    # Store output (Triton will cast to the dtype of out_ptr if needed)
    tl.store(out_row_ptr + cols * out_stride1, y_f32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == 4096, "This optimized kernel expects hidden size 4096."

        # Output tensor with same dtype and shape as input
        out = torch.empty_like(hidden)

        # Epsilon as in original code
        EPS = 1e-5

        # Strides (in elements)
        hidden_stride0 = hidden.stride(0)
        hidden_stride1 = hidden.stride(1)
        out_stride0 = out.stride(0)
        out_stride1 = out.stride(1)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, EPS,
            hidden_stride0, hidden_stride1,
            out_stride0, out_stride1,
            BLOCK_SIZE=4096,
            num_warps=8,   # best-performing in your environment
            num_stages=2,  # best-performing in your environment
        )

        return out


def run(*args):
    return ModelNew()(*args)
