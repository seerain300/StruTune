import torch
import triton
import triton.language as tl


@triton.jit
def divide_inplace_kernel(inp_ptr, out_ptr, n_elements: tl.constexpr, scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise division: out[i] = inp[i] / scale for i in [0, n_elements).
    Performs in-place division (out_ptr can be the same as inp_ptr).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    # scale is a scalar float; Triton will broadcast it. Ensure safe division.
    y = x / scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Fused padding and rFFT computation using Triton for normalization.
        Args:
            *args: Expected to be a single tensor x of shape (batch, channels, seqlen)
        Returns:
            x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1)
            x_freq_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1)
        """
        # Extract input
        assert len(args) == 1, "ModelNew.forward expects a single input tensor"
        x = args[0]
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Cast to float32 (original code does this)
        x_f32 = x.to(torch.float32)

        # Compute rFFT with implicit zero-padding to n=2*seqlen
        n = 2 * seqlen
        x_freq = torch.fft.rfft(x_f32, n=n)  # complex output of length n//2 + 1 = seqlen + 1

        # Extract real and imaginary parts
        x_freq_real = x_freq.real.contiguous()  # shape: (batch, channels, seqlen+1)
        x_freq_imag = x_freq.imag.contiguous()  # shape: (batch, channels, seqlen+1)

        # Normalize by n (2*seqlen). Use Triton kernels for division.
        total = batch * channels * (seqlen + 1)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total, BLOCK_SIZE),)

        # For real output
        x_freq_real_div = torch.empty_like(x_freq_real)
        divide_inplace_kernel[grid](x_freq_real, x_freq_real_div, total, n, BLOCK_SIZE=BLOCK_SIZE)

        # For imaginary output
        x_freq_imag_div = torch.empty_like(x_freq_imag)
        divide_inplace_kernel[grid](x_freq_imag, x_freq_imag_div, total, n, BLOCK_SIZE=BLOCK_SIZE)

        return x_freq_real_div, x_freq_imag_div


def run(*args):
    return ModelNew()(*args)
