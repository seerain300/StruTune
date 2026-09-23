import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Compute per-row inv_rms in float32, scale the row, multiply by weight (float32), store to output (original dtype).
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             batch_size, hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return

    row_offset = pid * hidden_size

    # Column offsets (0..BLOCK_SIZE-1), BLOCK_SIZE == hidden_size (4096)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # always true when hidden_size == BLOCK_SIZE

    # Load hidden row (original dtype), convert to float32 for computation
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute sum of squares and inv_rms (per-row)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)
    mean_sq = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Load weight vector and compute output: y = x * inv_rms * weight
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y_f32 = x_f32 * inv_rms * w

    # Store output (will cast to the dtype of out_ptr if needed)
    tl.store(out_ptr + row_offset + offs, y_f32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096 as per the original code."
        EPS = 1e-5

        # Output must match the original dtype
        out = torch.empty((batch_size, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out,
            batch_size, hidden_size, EPS,
            BLOCK_SIZE=hidden_size,
            num_warps=16,   # robust configuration for memory-bound kernels
            num_stages=2,   # pipeline stages
        )

        return out


def run(*args):
    return ModelNew()(*args)
