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

    # Compute pointers to the start of this row
    hidden_row_ptr = hidden_ptr + row_id * hidden_stride0
    out_row_ptr = out_ptr + row_id * out_stride0

    # Vector of column offsets for this row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # safety, though H == BLOCK_SIZE in this workload

    # Load the entire row (x may be fp16/bf16; cast to fp32 for math)
    x = tl.load(hidden_row_ptr + cols * hidden_stride1, mask=mask, other=0.0)
    x32 = tl.cast(x, tl.float32)

    # Compute sum of squares in fp32
    sumsq = tl.sum(x32 * x32, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to fp32
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    w32 = tl.cast(w, tl.float32)

    # Compute output in fp32: y = x * inv_rms * w
    y32 = x32 * inv_rms * w32

    # Store back; Triton will cast to the dtype of out_ptr if needed
    tl.store(out_row_ptr + cols * out_stride1, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are contiguous
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == 4096, "This Triton kernel is specialized for hidden size 4096."
        EPS = 1e-5

        # Allocate output with same shape and dtype as input
        out = torch.empty_like(hidden)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, EPS,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
