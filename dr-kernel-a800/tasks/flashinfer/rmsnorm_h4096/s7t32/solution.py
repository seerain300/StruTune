import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row (float32), compute inv_rms, and store it to a small buffer.
# - Pass 2: read inv_rms, scale, and write output (multiply by weight).
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, inv_rms_ptr, out_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * hidden_size

    # Vector of column offsets (0..BLOCK_SIZE-1), with BLOCK_SIZE == hidden_size
    offs = tl.arange(0, BLOCK_SIZE)
    # Mask isn't necessary when BLOCK_SIZE == hidden_size, but kept for generality
    mask = offs < hidden_size

    # Pass 1: accumulate sum of squares (float32)
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)

    # Compute inv_rms and store it
    inv_rms = tl.rsqrt(sum_sq / hidden_size + EPS)
    tl.store(inv_rms_ptr + pid, inv_rms)

    # Pass 2: scale and write output
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scaled = x_f32 * inv_rms * w
    tl.store(out_ptr + row_offset + offs, scaled, mask=mask)

# ModelNew entry point: Triton-only, no torch ops in host
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        # Shapes
        batch_size, hidden_size = hidden.shape
        assert hidden_size == 4096, "hidden_size must be 4096"

        # Compute in float32 for numerical stability
        hidden_f32 = hidden.to(torch.float32)
        w_f32 = w.to(torch.float32)

        # Allocate output (float32) and per-row inv_rms buffer
        out = torch.empty_like(hidden_f32)
        inv_rms = torch.empty(batch_size, device=hidden.device, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_f32, w_f32, inv_rms, out,
            hidden_size=hidden_size, EPS=1e-5, BLOCK_SIZE=hidden_size,
            num_warps=16, num_stages=2,
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
