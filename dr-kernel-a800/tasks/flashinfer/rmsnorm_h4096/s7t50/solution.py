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
    if pid >= batch_size:
        return

    row_base = pid * hidden_size

    # First pass: compute sum of squares across the row in float32
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # always true when BLOCK_SIZE == hidden_size
    x = tl.load(hidden_ptr + row_base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    sumsq = tl.sum(x32 * x32, axis=0)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Second pass: scale and store output, multiplying by weight
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x32 * inv_rms * w
    tl.store(out_ptr + row_base + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure tensors are CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        EPS = 1e-5

        # Output tensor with same dtype as input
        out = torch.empty_like(hidden)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        # Tuned launch parameters: 32 warps often yield better throughput for memory-bound ops
        _row_fused_scale_kernel[grid](
            hidden, weight, out,
            batch_size, hidden_size, EPS,
            BLOCK_SIZE=hidden_size,
            num_warps=32,  # tuned for performance
            num_stages=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
