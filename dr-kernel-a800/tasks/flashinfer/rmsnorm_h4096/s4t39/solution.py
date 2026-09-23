import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (we'll load and cast to float32 in-kernel)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    # One program per row
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # mask for generality; with H=4096 and BLOCK_SIZE=4096 this is all true

    # Load the entire row (masked), cast to float32 for math
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0 + cols * hidden_stride1
    x = tl.load(hidden_row_ptr, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute mean of squares and inv_rms
    sumsq = tl.sum(x * x, axis=0)
    H_inv = 1.0 / H
    mean = sumsq * H_inv
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector as float32
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    weight = tl.cast(weight, tl.float32)

    # Compute output: y = x * inv_rms * weight
    y = x * inv_rms * weight

    # Store result
    out_row_ptr = out_ptr + row_id * out_stride0 + cols * out_stride1
    tl.store(out_row_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "ModelNew requires CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == 4096, "Expected hidden_size == 4096"

        EPS = 1e-5

        # Allocate output with same dtype and shape as input
        out = torch.empty_like(hidden)

        # Compute strides in element units
        hidden_stride0 = hidden.stride(0)
        hidden_stride1 = hidden.stride(1)
        out_stride0 = out.stride(0)
        out_stride1 = out.stride(1)

        # Launch grid: one program per row
        grid = (B,)

        # Use the proven best configuration for this workload
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
