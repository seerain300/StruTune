import torch
import triton
import triton.language as tl

# Triton kernel: จัดการแถวเดียวต่อ program
# For each row b:
#   - Load x[b, :] (cast to float32),
#   - Compute inv_rms = rsqrt(mean(x[b, :]**2) + EPS),
#   - Scale and store y[b, :] = x[b, :] * inv_rms * weight.
@triton.jit
def _row_fused_scale_singlepass_kernel(
    hidden_ptr,           # *ptr to hidden_states (original dtype, e.g., bfloat16)
    weight_ptr,           # *ptr to weight (original dtype, e.g., bfloat16)
    out_ptr,              # *ptr to output (same dtype as hidden_states)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,        # 4096 in this model
    EPS: tl.constexpr,                # 1e-5
    BLOCK_SIZE: tl.constexpr          # 4096
):
    pid = tl.program_id(axis=0)  # one program per row
    base = pid * hidden_size

    # Load the entire row into float32
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # always true for hidden_size=4096
    x = tl.load(hidden_ptr + base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute inv_rms
    sumsq = tl.sum(x32 * x32, axis=0)  # scalar
    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)     # scalar

    # Load weight vector, convert to float32, scale
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y32 = x32 * inv_rms * w

    # Store as output dtype (out_ptr type defines dtype)
    tl.store(out_ptr + base + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # If not on CUDA, fall back to PyTorch implementation for correctness
        if (not hidden_states.is_cuda) or (not weight.is_cuda):
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure tensors are contiguous
        hidden = hidden_states.contiguous()
        w = weight.contiguous()
        batch_size = hidden.shape[0]
        hidden_size = hidden.shape[1]
        assert hidden_size == 4096, "This Triton kernel expects hidden_size == 4096"

        # Output tensor in the same dtype as hidden_states
        out = torch.empty_like(hidden)

        # Choose launch config
        BLOCK_SIZE = 4096
        grid = (batch_size,)
        # Use a high-occupancy setting for memory-bound kernels
        _row_fused_scale_singlepass_kernel[grid](
            hidden, w, out,
            batch_size=batch_size,
            hidden_size=hidden_size,
            EPS=1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
