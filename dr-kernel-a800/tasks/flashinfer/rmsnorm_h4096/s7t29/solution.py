import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Assumes hidden_size == 4096. Single vector pass for sum of squares, single vector pass for scaling/store.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * hidden_size

    # Vector of column offsets (0..BLOCK_SIZE-1), with BLOCK_SIZE == hidden_size
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # mask is always true for BLOCK_SIZE == hidden_size, but kept for safety

    # Pass 1: accumulate sum of squares across the row (float32)
    sum_sq = 0.0
    # Since hidden_size == BLOCK_SIZE, we can load the entire row in one go
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)

    # Compute inv_rms
    inv_rms = tl.rsqrt(sum_sq / hidden_size + EPS)

    # Pass 2: scale and store
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y_f32 = x_f32 * inv_rms * w
    # Store back; Triton will infer dtype from out_ptr (we'll allocate out as the original dtype)
    tl.store(out_ptr + row_offset + offs, y_f32, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernel"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        # Output tensor with same dtype and shape as hidden_states
        out = torch.empty_like(hidden_states)

        # Constants
        hidden_size = hidden_states.shape[-1]
        assert hidden_size == 4096, "This Triton kernel assumes hidden_size == 4096"

        # Launch: one program per row
        grid = (hidden_states.shape[0],)
        # Use robust defaults; for 4096-wide rows, num_warps=8 and num_stages=1 work well across GPUs
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out,
            hidden_size=hidden_size, EPS=1e-5, BLOCK_SIZE=hidden_size,
            num_warps=8, num_stages=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
