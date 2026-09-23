import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms.
# - Pass 2: scale and store output, multiplying by weight.
@triton.jit
def _row_fused_scale_kernel(
    hidden_ptr,       # *pointer* to input hidden states (original dtype), assumed contiguous
    weight_ptr,       # *pointer* to weight (original dtype), 1D
    out_ptr,          # *pointer* to output (float32), contiguous
    batch_size,       # int: number of rows
    EPS,              # float32 epsilon
    BLOCK_SIZE: tl.constexpr,  # number of columns per row (here 4096)
):
    pid = tl.program_id(axis=0)  # row id
    if pid >= batch_size:
        return

    row_off = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute sum of squares and inv_rms
    sum_sq = 0.0
    x = tl.load(hidden_ptr + row_off + offs)
    x_f32 = x.to(tl.float32)
    sum_sq += tl.sum(x_f32 * x_f32)
    mean = sum_sq / BLOCK_SIZE
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and store
    w = tl.load(weight_ptr)  # scalar weight (same dtype as weight tensor)
    y = (x_f32 * inv_rms) * w
    tl.store(out_ptr + row_off + offs, y)  # store as float32

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        # Shapes
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Make inputs contiguous for predictable layout
        hidden_states_c = hidden_states.contiguous()
        weight_c = weight.contiguous()

        # Compute in float32 inside the kernel
        out_f32 = torch.empty((batch_size, hidden_size), device=hidden_states.device, dtype=torch.float32)
        EPS = 1e-5

        # Launch one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_states_c, weight_c, out_f32,
            batch_size,
            EPS,
            BLOCK_SIZE=4096,
            num_warps=16,
            num_stages=1,
        )

        # Cast output back to the original dtype to match the original model's behavior
        # The original PyTorch code returns y.to(hidden_states.dtype), so we do that here.
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
