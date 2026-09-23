import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const float32 (host provides weight as float32)
    out_ptr,          # *output (same dtype as hidden, e.g., bfloat16)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon
    BLOCK_SIZE: tl.constexpr,  # we set this to H (4096) for full-row vectorization
):
    # One Triton program per row
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    cols = tl.arange(0, BLOCK_SIZE)  # vector of column indices [0..H-1]

    # Compute base pointers for this row (row-major contiguous tensors)
    hidden_row_ptr = hidden_ptr + row_id * H + cols
    out_row_ptr = out_ptr + row_id * H + cols

    # Load hidden row and cast to float32 for math
    x = tl.load(hidden_row_ptr)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)  # scalar
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector as float32 and compute output
    w = tl.load(weight_ptr + cols)  # weight_ptr is float32
    y = x * inv_rms * w  # float32 math

    # Store output; Triton will cast to out_ptr's element dtype if needed
    tl.store(out_row_ptr, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
        hidden = hidden_states.contiguous()
        # weight is [H], cast to float32 for numerically stable math
        weight_f32 = weight.contiguous().to(torch.float32)

        B, H = hidden.shape
        EPS = 1e-5

        # Allocate output with the same dtype as hidden_states
        out = torch.empty_like(hidden)

        # Launch one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight_f32, out,
            B, H, EPS,
            BLOCK_SIZE=H,  # process the entire row in one vector
            num_warps=8,   # empirically best for this workload
            num_stages=2   # good default
        )

        return out


def run(*args):
    return ModelNew()(*args)
