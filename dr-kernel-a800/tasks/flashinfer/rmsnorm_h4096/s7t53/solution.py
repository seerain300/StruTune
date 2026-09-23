import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms.
# - Pass 2: scale and store output, multiplying by weight.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             batch_size, hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute sum of squares and inv_rms
    sumsq = tl.zeros((), dtype=tl.float32)
    for start in range(0, hidden_size, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hidden_size
        x = tl.load(hidden_ptr + pid * hidden_size + idx, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and store output
    for start in range(0, hidden_size, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hidden_size
        x = tl.load(hidden_ptr + pid * hidden_size + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + pid * hidden_size + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are CUDA tensors and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096."

        # Output tensor in float32 (compute precision), cast back to original dtype after
        out_fp32 = torch.empty_like(hidden_states, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out_fp32,
            batch_size=batch_size,
            hidden_size=hidden_size,
            EPS=1e-5,
            BLOCK_SIZE=4096,
            num_warps=16,
            num_stages=1,
        )

        # Cast back to original dtype to match original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
