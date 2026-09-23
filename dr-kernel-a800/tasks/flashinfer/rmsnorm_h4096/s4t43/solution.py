import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight
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

    # Column offsets for the entire row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # for H=4096 this is all-true; keeps generality

    # Row base pointers
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0
    out_row_ptr = out_ptr + row_id * out_stride0

    # Load the entire row and cast to float32 for computation
    x = tl.load(hidden_row_ptr + offs * hidden_stride1, mask=mask, other=0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector (cast to float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0)  # weight is [H]
    w = tl.cast(w, tl.float32)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store result to output (dtype of out_ptr determines output dtype)
    tl.store(out_row_ptr + offs * out_stride1, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are CUDA tensors and contiguous
        if not hidden_states.is_cuda or not weight.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors.")
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        EPS = 1e-5

        # Allocate output tensor with same shape and dtype as input
        out = torch.empty_like(hidden_states)

        # Launch kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden_states, weight, out,
            B, H, EPS,
            hidden_states.stride(0), hidden_states.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
