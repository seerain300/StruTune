import torch
import triton
import triton.language as tl


@triton.jit
def normalize_real_div_kernel(real_ptr, out_ptr, n_elements, scale):
    """
    Elementwise division of real output by scale.
    """
    i = tl.program_id(axis=0) * 1024 + tl.arange(0, 1024)
    mask = i < n_elements
    x = tl.load(real_ptr + i, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + i, x, mask=mask)


@triton.jit
def normalize_imag_div_kernel(imag_ptr, out_ptr, n_elements, scale):
    """
    Elementwise division of imaginary output by scale.
    """
    i = tl.program_id(axis=0) * 1024 + tl.arange(0, 1024)
    mask = i < n_elements
    x = tl.load(imag_ptr + i, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + i, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-normalized version of the original run function:
        - Compute torch.fft.rfft(x, n=2*seqlen) to ensure correctness.
        - Normalize real and imaginary parts by 2*seqlen using Triton kernels.
        - Return (batch, channels, seqlen+1) outputs with real and imag tensors.
        """
        # Assume input is a single tensor x with shape (batch, channels, seqlen)
        x = args[0]
        batch, channels, seqlen = x.shape

        # Cast to float32 for FFT numerical stability
        x_f32 = x.to(torch.float32)

        # Compute rFFT on the last dimension (seqlen), pad to 2*seqlen
        n = 2 * seqlen
        x_freq = torch.fft.rfft(x_f32, n=n)  # complex output, length n//2 + 1 = seqlen + 1

        # We need to extract real and imag parts, normalize, and return tensors.
        # PyTorch keeps complex dtype, so we use .real and .imag to get float32.
        real_part = x_freq.real
        imag_part = x_freq.imag

        # Ensure contiguous memory for Triton kernels
        real_part = real_part.contiguous()
        imag_part = imag_part.contiguous()

        # Prepare output buffers
        out_real = torch.empty_like(real_part)  # (batch, channels, seqlen+1)
        out_imag = torch.empty_like(imag_part)  # (batch, channels, seqlen+1)

        # Total number of elements (flattened) for division
        n_elements = real_part.numel()
        scale = float(n)  # 2 * seqlen

        # Launch Triton normalization kernels
        BLOCK = 1024
        grid_real = (triton.cdiv(n_elements, BLOCK),)
        grid_imag = (triton.cdiv(n_elements, BLOCK),)

        normalize_real_div_kernel[grid_real](real_part, out_real, n_elements, scale, num_warps=4)
        normalize_imag_div_kernel[grid_imag](imag_part, out_imag, n_elements, scale, num_warps=4)

        # Reshape outputs back to (batch, channels, seqlen + 1)
        out_real = out_real.view(batch, channels, seqlen + 1)
        out_imag = out_imag.view(batch, channels, seqlen + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
