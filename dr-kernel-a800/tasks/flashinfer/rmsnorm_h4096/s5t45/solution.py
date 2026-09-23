import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096  # fixed by original assert
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_out, EPS,
                        BLOCK_SIZE: tl.constexpr):
    # One Triton program per row
    row = tl.program_id(0)

    # Vectorized offsets across the row (compile-time constant)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # for H=4096, always true; kept for generality

    # Load the entire row (masked), cast to float32
    x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inverse RMS per row: rsqrt(mean + EPS)
    inv_rms = tl.rsqrt(sumsq / H + EPS)

    # Load weight vector (masked), cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute output: y32 = x32 * inv_rms * w32
    y32 = x32 * inv_rms * w

    # Store output (FP32) for this row
    tl.store(out_ptr + row * stride_out + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = hidden.shape

        # Output buffer in FP32; we'll cast to original dtype at the end
        out = torch.empty((B, H), device=hidden.device, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_scale_row[grid](
            hidden, w, out,
            B, H, hidden.stride(0), out.stride(0), EPS,
            BLOCK_SIZE=HIDDEN_SIZE  # must be tl.constexpr
        )

        # Match original behavior: return in the original hidden_states dtype
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
