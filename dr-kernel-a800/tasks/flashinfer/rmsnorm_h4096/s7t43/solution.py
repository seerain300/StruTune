import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms.
# - Pass 2: scale and store output, multiplying by weight.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             batch_size: tl.constexpr, hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    # Vector of column indices for the row
    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute sum of squares in float32
    x = tl.load(hidden_ptr + pid * hidden_size + offs)  # load original dtype
    x32 = x.to(tl.float32)
    sumsq = tl.sum(x32 * x32)
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Pass 2: scale and store output
    w = tl.load(weight_ptr + offs).to(tl.float32)          # weight is float32 in this workload
    y32 = x32 * inv_rms * w
    tl.store(out_ptr + pid * hidden_size + offs, y32)     # Triton will cast to out_ptr element type if needed

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden_states.shape
        # Output in the same dtype as input
        out = torch.empty_like(hidden_states)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        # Fixed hidden_size for this problem is 4096; BLOCK_SIZE must match
        _row_fused_scale_kernel[grid](
            hidden_states, weight, out,
            batch_size=batch_size, hidden_size=hidden_size, EPS=1e-5,
            BLOCK_SIZE=4096, num_warps=16, num_stages=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
