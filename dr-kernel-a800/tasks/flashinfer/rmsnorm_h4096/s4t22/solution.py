import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (float32, host provides)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (expected 4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    # Base pointers for this row
    row_hidden_ptr = hidden_ptr + row_id * hidden_stride0
    row_out_ptr = out_ptr + row_id * out_stride0

    # Column offsets
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H  # robust if H != BLOCK_SIZE; here H == 4096

    # Load x row and cast to float32 for math
    x = tl.load(row_hidden_ptr + cols * hidden_stride1, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares and inv_rms
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector (float32) and compute y
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # weight_ptr is float32
    y = x * inv_rms * w

    # Store result
    tl.store(row_out_ptr + cols * out_stride1, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        # Convert weight to float32 for math, keep contiguous
        weight = weight.to(torch.float32).contiguous()

        B, H = hidden_states.shape
        # Output matches input dtype
        out = torch.empty_like(hidden_states)

        # Strides in elements
        hidden_stride0 = hidden_states.stride(0)
        hidden_stride1 = hidden_states.stride(1)
        out_stride0 = out.stride(0)
        out_stride1 = out.stride(1)

        # Launch Triton kernel with standard grid and meta-parameters
        EPS = 1e-5
        _layernorm_weight_scale_kernel[(B,)](
            hidden_states, weight, out,
            B, H, EPS,
            hidden_stride0, hidden_stride1, out_stride0, out_stride1,
            BLOCK_SIZE=4096,
            num_warps=8,  # best-performing in your environment
            num_stages=2  # best-performing in your environment
        )
        return out


def run(*args):
    return ModelNew()(*args)
