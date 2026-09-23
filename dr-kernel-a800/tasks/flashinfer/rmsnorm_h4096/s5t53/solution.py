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

    # Vector of column offsets (constexpr)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # H == BLOCK_SIZE, but keep mask for safety

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the entire row (hidden_states), cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum of squares across the row (scalar)
    sumsq = tl.sum(x * x, axis=0)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Load weight vector, cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute scaled output
    y32 = x * inv_rms * w

    # Store result (float32; host will cast to desired dtype)
    tl.store(out_row_ptr + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, "hidden_states must have 4096 columns."

        # Allocate output as float32 for computation precision
        out = torch.empty((B, H), device=hidden.device, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_scale_row[grid](
            hidden, weight, out,
            B, H, hidden.stride(0), out.stride(0),
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=8,  # good occupancy for 4096-wide vector
            num_stages=1
        )

        # Match original behavior: return in original hidden_states dtype
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
