import torch
import triton
import triton.language as tl


@triton.jit
def normalize_divide_kernel(in_ptr, out_ptr, n_elements: tl.constexpr, scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise normalization: out[i] = in[i] / scale.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # scale is a Python float; Triton will treat it as a scalar constant.
    vals = vals / scale
    tl.store(out_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Computes rFFT of x using PyTorch (torch.fft.rfft) and normalizes via Triton.
        Returns real and imaginary parts of shape (batch, channels, seqlen + 1).
        """
        assert x.ndim == 3, "Input must be of shape (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Compute rFFT using PyTorch to ensure correctness across all axes
        x_f32 = x.to(torch.float32)
        y = torch.fft.rfft(x_f32, n=N)  # complex output per (batch, channel)

        # Separate real and imaginary parts
        y_real = y.real.contiguous()
        y_imag = y.imag.contiguous()

        # Normalize by N = 2 * seqlen. Use Triton to perform elementwise division.
        # Flatten to 1D for simplicity
        real_flat = y_real.view(-1)
        imag_flat = y_imag.view(-1)

        n_real = real_flat.numel()
        n_imag = imag_flat.numel()
        scale = float(N)  # normalization factor

        # Triton normalization: out = in / scale
        # Choose a reasonable block size; 1024 works well for typical sizes.
        BLOCK_SIZE = 1024
        grid_real = (triton.cdiv(n_real, BLOCK_SIZE),)
        grid_imag = (triton.cdiv(n_imag, BLOCK_SIZE),)

        real_out = torch.empty_like(real_flat)
        imag_out = torch.empty_like(imag_flat)

        normalize_divide_kernel[grid_real](real_flat, real_out, n_real, scale, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        normalize_divide_kernel[grid_imag](imag_flat, imag_out, n_imag, scale, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Reshape back to (batch, channels, seqlen + 1)
        y_real = real_out.view(batch, channels, seqlen + 1)
        y_imag = imag_out.view(batch, channels, seqlen + 1)
        return y_real, y_imag


def run(*args):
    return ModelNew()(*args)
