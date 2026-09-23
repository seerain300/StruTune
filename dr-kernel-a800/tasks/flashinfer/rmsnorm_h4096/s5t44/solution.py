import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_out, EPS):
    # One program per row
    row = tl.program_id(0)
    offs = tl.arange(0, H)
    mask = offs < H  # always true for H=4096; keep for safety

    # Load hidden row, cast to float32
    hidden_row_ptr = hidden_ptr + row * stride_hs
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute sum of squares (masked lanes are zero)
    x_sq = x32 * x32
    sumsq = tl.sum(x_sq, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + EPS)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector, cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w32 = w.to(tl.float32)

    # Apply normalization and scale
    y32 = x32 * inv_rms * w32

    # Store output (FP32)
    out_row_ptr = out_ptr + row * stride_out
    tl.store(out_row_ptr + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden_size == {HIDDEN_SIZE}, got {H}"

        # Allocate FP32 output buffer
        out = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch kernel: one program per row
        grid = (B,)
        normalize_scale_row[grid](
            hidden, weight, out,
            B, H, hidden.stride(0), out.stride(0), EPS
        )

        # Cast to original hidden dtype to match original model behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
