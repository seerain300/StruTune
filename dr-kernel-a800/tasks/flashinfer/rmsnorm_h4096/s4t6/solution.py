import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const float32 (converted to float32 on host)
    out_ptr,          # *output (same dtype as hidden, e.g., bfloat16)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # must be H=4096 for single-chunk processing
):
    # One program per row
    row = tl.program_id(0)

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * hidden_stride0
    out_row_ptr = out_ptr + row * out_stride0

    # Vector of column offsets for this row
    offs = tl.arange(0, BLOCK_SIZE)

    # Load entire row of hidden states, cast to float32 explicitly
    x = tl.load(hidden_row_ptr + offs * hidden_stride1)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares and mean
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # scalar float32

    # Load weight vector as float32
    w = tl.load(weight_ptr + offs)  # weight_ptr is float32

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w  # all math in float32

    # Store result; Triton will cast to the dtype of out_ptr if needed
    tl.store(out_row_ptr + offs * out_stride1, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B, H = hidden_states.shape
        assert H == weight.shape[0], f"Weight size {weight.shape[0]} must match hidden_states last dim {H}."
        # For this optimized kernel, hidden size is fixed at 4096
        assert H == 4096, "This optimized Triton kernel expects hidden size 4096."

        # Make inputs contiguous for performance
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Output tensor with same shape and dtype as hidden_states
        out = torch.empty_like(hidden)

        # Strides (in elements)
        hidden_stride0 = hidden.stride(0)
        hidden_stride1 = hidden.stride(1)
        out_stride0 = out.stride(0)
        out_stride1 = out.stride(1)

        EPS = 1e-5

        # Kernel launch configuration: process entire row in one chunk
        BLOCK_SIZE = 4096
        grid = (B,)

        _layernorm_weight_scale_kernel[grid](
            hidden, weight.float(), out,
            B, H, EPS,
            hidden_stride0, hidden_stride1,
            out_stride0, out_stride1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,  # higher warp count can improve throughput on wide vectors
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
