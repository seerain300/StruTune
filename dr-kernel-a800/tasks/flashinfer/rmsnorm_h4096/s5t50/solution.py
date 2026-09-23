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
    # H == BLOCK_SIZE; mask is all True, but kept for safety
    mask = offs < H

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the entire row (hidden_states) and cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute output: y32 = x32 * inv_rms * w32
    y = x * inv_rms * w

    # Store result
    tl.store(out_row_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure CUDA tensors and contiguous layout
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden size 4096, got {H}"

        # Allocate output in float32
        out_fp32 = torch.empty_like(hidden, dtype=torch.float32)

        # Strides (in elements)
        stride_hs = hidden.stride(0)
        stride_out = out_fp32.stride(0)

        # Launch Triton kernel: one program per row, BLOCK_SIZE=4096
        grid = (B,)
        normalize_scale_row[grid](
            hidden, weight, out_fp32,
            B, H, stride_hs, stride_out,
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=8,  # reasonable for 4096-wide vector
            num_stages=1
        )

        # Return with original dtype to match original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
