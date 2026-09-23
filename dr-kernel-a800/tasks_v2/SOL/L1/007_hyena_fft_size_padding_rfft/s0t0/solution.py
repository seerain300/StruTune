import torch
import triton
import triton.language as tl


@triton.jit
def divide_by_scalar_kernel(in_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: elementwise divide each element in 'in_ptr' by 'scale' and write to 'out_ptr'.
    Operates on a flattened 1D view of the input tensor. Assumes float32.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + offsets, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape (batch, channels, seqlen)
        if len(args) == 0:
            raise ValueError("No input provided to ModelNew.forward")
        x = args[0]

        # Ensure float32 for numerical stability (original code casts to float32)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if x.dim() != 3:
            raise ValueError(f"Expected input with 3 dimensions (batch, channels, seqlen), got shape {tuple(x.shape)}")
        batch, channels, seqlen = x.shape

        # Perform real FFT with n = 2 * seqlen (matches original code)
        # Output is complex and has shape (batch, channels, seqlen + 1)
        x_freq = torch.fft.rfft(x, n=2 * seqlen)

        # Extract real and imaginary parts
        x_freq_real = x_freq.real.contiguous()  # (B, C, seqlen+1)
        x_freq_imag = x_freq.imag.contiguous()  # (B, C, seqlen+1)

        # Allocate outputs for normalized real/imag parts
        out_real = torch.empty_like(x_freq_real)
        out_imag = torch.empty_like(x_freq_imag)

        # Flatten for 1D Triton processing
        n_real = x_freq_real.numel()
        n_imag = x_freq_imag.numel()

        # Use a larger block size to reduce launch count; 2048 is a good default
        BLOCK_SIZE = 2048
        grid_real = (triton.cdiv(n_real, BLOCK_SIZE),)
        grid_imag = (triton.cdiv(n_imag, BLOCK_SIZE),)

        # Normalization factor: 1 / (2 * seqlen)
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernels to perform the elementwise division
        divide_by_scalar_kernel[grid_real](x_freq_real, out_real, n_real, scale, BLOCK_SIZE=BLOCK_SIZE)
        divide_by_scalar_kernel[grid_imag](x_freq_imag, out_imag, n_imag, scale, BLOCK_SIZE=BLOCK_SIZE)

        # Return real and imaginary parts (two separate tensors), matching original signature
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
