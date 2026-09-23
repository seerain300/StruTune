import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_row_kernel(hidden_ptr, weight_ptr, out_ptr,
                                B, H,
                                BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: one program per row.
    hidden_ptr: *ptr to [B, H], float32
    weight_ptr: *ptr to [H], float32
    out_ptr: *ptr to [B, H], float32
    B, H: runtime ints
    BLOCK_SIZE: compile-time constant, set to HIDDEN_SIZE
    """
    row = tl.program_id(axis=0)
    if row >= B:
        return

    # Vectorized offsets across the row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # true for all lanes when BLOCK_SIZE == H

    # Load the entire row as float32
    x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)

    # Compute inv_rms: rsqrt(mean + EPS)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Load weight vector (already float32), compute output
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    y = x * inv_rms * w

    # Store result
    tl.store(out_ptr + row * H + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden size {HIDDEN_SIZE}, got {H}"

        # Allocate output in float32 (compute dtype)
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _normalize_scale_row_kernel[grid](
            hidden.float(),          # hidden_ptr: convert to fp32 for computation
            weight.float(),          # weight_ptr: convert to fp32 for computation
            out_fp32,                # out_ptr
            B, H,
            BLOCK_SIZE=HIDDEN_SIZE   # tl.constexpr
        )

        # Cast to original hidden_states dtype to match PyTorch behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
