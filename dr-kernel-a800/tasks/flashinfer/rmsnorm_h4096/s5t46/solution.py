import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_out):
    # One program per row
    row = tl.program_id(0)

    # Vector of column offsets (compile-time constant for Triton)
    offs = tl.arange(0, H)
    mask = offs < H  # all True when H=4096

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the row and cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inverse RMS: rsqrt(mean(x^2) + EPS)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w32 = w.to(tl.float32)

    # Scale and store result
    y32 = x32 * inv_rms * w32
    tl.store(out_row_ptr + offs, y32, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguous layout
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        # Output in fp32 during compute; cast to original dtype at the end
        out = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row, with BLOCK_SIZE=H (compile-time)
        grid = (B,)
        normalize_scale_row[grid](
            hidden, weight, out,
            B, H, hidden.stride(0), out.stride(0),
            BLOCK_SIZE=H,  # tl.constexpr for arange
            num_warps=8,   # tuning for performance
            num_stages=2
        )

        # Match original behavior: cast back to hidden_states dtype
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
