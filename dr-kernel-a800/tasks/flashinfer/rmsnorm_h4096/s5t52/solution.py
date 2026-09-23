import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_out,
                        BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)

    # Vector of column offsets (constexpr for Triton)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # H == BLOCK_SIZE, so all lanes are valid; mask kept for safety

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the entire row (hidden_states), cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)           # scalar
    mean_sq = sumsq / H                      # scalar
    inv_rms = tl.rsqrt(mean_sq + EPS)       # scalar

    # Load weight vector, cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w = w.to(tl.float32)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store output (float32)
    tl.store(out_row_ptr + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden_size == {HIDDEN_SIZE}, got {H}"

        # Output buffer in float32 for computation stability
        out = torch.empty((B, H), device=hidden.device, dtype=torch.float32)

        # Launch Triton kernel: one program per row, BLOCK_SIZE=H
        grid = (B,)
        normalize_scale_row[grid](
            hidden, weight, out,
            B, H, hidden.stride(0), out.stride(0),
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=8,  # good occupancy for a 4096-wide vector
            num_stages=1
        )

        # Cast back to the original hidden_states dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
