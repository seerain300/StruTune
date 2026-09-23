import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_row_kernel(hidden_ptr, weight_ptr, out_ptr,
                                B, H,
                                BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    # If row >= B, exit (guard in case grid > B)
    if row >= B:
        return

    # Pass 1: compute sum of squares across the row
    sumsq = 0.0
    for c in range(0, H, BLOCK_SIZE):
        offs = c + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load row segment as float32
        x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    # Compute inv_rms = rsqrt(mean + EPS)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Pass 2: apply normalization and weight, store FP32
    for c in range(0, H, BLOCK_SIZE):
        offs = c + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + row * H + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        # Allocate output as float32 for computation
        out = torch.empty((B, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _normalize_scale_row_kernel[grid](
            hidden_states, weight, out,
            B, H,
            BLOCK_SIZE=1024,  # compile-time chunk size
            num_warps=4,      # tuning knobs; 4 or 8 typically fine
            num_stages=2
        )

        # Return in original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
