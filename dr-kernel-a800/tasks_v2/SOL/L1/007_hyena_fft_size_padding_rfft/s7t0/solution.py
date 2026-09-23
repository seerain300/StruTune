import torch
import triton
import triton.language as tl


@triton.jit
def _copy_and_scale_real_kernel(inp_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    # Normalize in-kernel: y = x * scale
    y = x * scale
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _copy_and_scale_imag_kernel(inp_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    y = x * scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect a single 3D tensor: (batch, channels, seqlen)
        if x.dim() != 3:
            raise ValueError(f"ModelNew expects a 3D tensor (batch, channels, seqlen). Got shape {tuple(x.shape)}")

        batch, channels, seqlen = x.shape

        # Cast to float32 (original code does this)
        x_f32 = x.to(torch.float32)

        # Compute real FFT with zero padding to 2 * seqlen
        # Output shape: (batch, channels, seqlen + 1)
        y_complex = torch.fft.rfft(x_f32, n=2 * seqlen)

        # Normalize by 2 * seqlen (same as original)
        norm = 2.0 * seqlen
        y_complex = y_complex / norm

        # Allocate outputs for real and imaginary parts
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # We'll copy real/imag parts using Triton and apply scaling in-kernel
        # Flatten views for 1D Triton copying
        total_real = y_complex.real.numel()
        total_imag = y_complex.imag.numel()

        # Choose a block size; 4096 is a good default for 1D copies
        BLOCK_SIZE = 4096
        grid_real = (triton.cdiv(total_real, BLOCK_SIZE),)
        grid_imag = (triton.cdiv(total_imag, BLOCK_SIZE),)

        # Scale factor is 1 / norm
        scale = 1.0 / norm

        _copy_and_scale_real_kernel[grid_real](
            y_complex.real, x_freq_real, total_real, scale, BLOCK_SIZE=BLOCK_SIZE,
        )
        _copy_and_scale_imag_kernel[grid_imag](
            y_complex.imag, x_freq_imag, total_imag, scale, BLOCK_SIZE=BLOCK_SIZE,
        )

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
