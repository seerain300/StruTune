import torch
import triton
import triton.language as tl

# Constants
HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_w, stride_out,
                        BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)

    # Vector of column offsets (compile-time constant for Triton)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # safety; BLOCK_SIZE == H, mask all True

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the entire row (hidden_states), cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Scale: x32 * inv_rms
    x_scaled = x * inv_rms

    # Load weight vector, cast to float32
    w = tl.load(weight_ptr + offs * stride_w, mask=mask, other=0.0).to(tl.float32)

    # Final output: scaled * weight
    y = x_scaled * w

    # Store result
    tl.store(out_row_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        # Output in float32 for numerical stability, then cast back at the end
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        # Strides are in elements (PyTorch gives strides in elements)
        stride_hs = hidden.stride(0)
        stride_out = out_fp32.stride(0)
        # weight is 1D, contiguous, stride_w = 1
        stride_w = weight.stride(0)

        # Use more warps to improve throughput on 4096-wide vectors
        normalize_scale_row[grid](
            hidden, weight, out_fp32,
            B, H, stride_hs, stride_w, stride_out,
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=16,  # tuned for 4096-wide vector
            num_stages=1
        )

        # Cast back to original dtype to match original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
