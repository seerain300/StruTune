import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_row_kernel(hidden_ptr, weight_ptr, out_ptr,
                                B, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= B:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # true for all lanes when BLOCK_SIZE == H

    # Load the entire row as float32
    x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)

    # Compute inv_rms: rsqrt(mean + EPS)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Load weight vector as float32 and apply scaling
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w  # FP32 compute

    # Store FP32 output
    tl.store(out_ptr + row * H + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        if not hidden_states.is_cuda or not weight.is_cuda:
            raise RuntimeError("hidden_states and weight must be CUDA tensors")
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        # Allocate FP32 output
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _normalize_scale_row_kernel[grid](
            hidden_states, weight, out_fp32,
            B=B, H=H, BLOCK_SIZE=H
        )

        # Cast back to original dtype to match PyTorch behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
