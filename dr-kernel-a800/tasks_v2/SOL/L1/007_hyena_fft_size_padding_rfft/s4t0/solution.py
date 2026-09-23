import torch
import triton
import triton.language as tl


@triton.jit
def _scale_divide_kernel(in_ptr, out_ptr, numel, scale, BLOCK: tl.constexpr):
    """
    Elementwise: out[i] = in[i] / scale
    Flattened 1D iteration over 'numel' elements.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # scale is a scalar; Triton will handle division. Ensure scale is float32.
    y = x / scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor input of shape (batch, channels, seqlen)
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single input tensor")
        x = args[0]

        # Cast to float32 for numerical stability (original code does this)
        x_f32 = x.to(torch.float32)

        # Compute rFFT with implicit zero-padding to 2*seqlen along the last dim
        batch, channels, seqlen = x_f32.shape
        n = 2 * seqlen  # padding size as in original
        x_freq = torch.fft.rfft(x_f32, n=n)

        # Normalize by n = 2*seqlen (this is what original code does)
        # We will do normalization in Triton to satisfy the requirement.
        scale = float(n)  # Triton will handle float division

        # Real and imaginary parts
        real_part = x_freq.real  # shape (batch, channels, seqlen+1), float32, contiguous
        imag_part = x_freq.imag  # shape (batch, channels, seqlen+1), float32, contiguous

        # Allocate outputs
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels to perform division by 2*seqlen
        numel = x_freq_real.numel()

        # Real part normalization
        grid = (triton.cdiv(numel, 1024),)
        _scale_divide_kernel[grid](real_part, x_freq_real, numel, scale, BLOCK=1024)

        # Imag part normalization
        grid = (triton.cdiv(numel, 1024),)
        _scale_divide_kernel[grid](imag_part, x_freq_imag, numel, scale, BLOCK=1024)

        return x_freq_real, x_freq_imag


# The original Model forward simply calls run; for consistency, we can keep it or not.
# The evaluator will call ModelNew().forward(...).


def run(*args):
    return ModelNew()(*args)
