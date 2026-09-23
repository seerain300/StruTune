import torch
import triton
import triton.language as tl


@triton.jit
def rfft_divide_kernel(input_real_ptr, input_imag_ptr,
                       output_real_ptr, output_imag_ptr,
                       N_total, norm, BLOCK: tl.constexpr):
    """
    Elementwise normalize real and imaginary parts by 'norm'.
    Reads from input_real_ptr/input_imag_ptr (flattened),
    writes to output_real_ptr/output_imag_ptr (flattened).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_total

    # Load real and imaginary parts
    v_real = tl.load(input_real_ptr + offsets, mask=mask, other=0.0)
    v_imag = tl.load(input_imag_ptr + offsets, mask=mask, other=0.0)

    # Normalize by 2 * seqlen (provided as 'norm')
    v_real = v_real / norm
    v_imag = v_imag / norm

    # Store results
    tl.store(output_real_ptr + offsets, v_real, mask=mask)
    tl.store(output_imag_ptr + offsets, v_imag, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute real FFT (rfft) of x with implicit zero-padding to 2*seqlen,
        normalize by 2*seqlen, and return real and imaginary parts separately.
        Entry point: ModelNew
        """
        # Ensure float32 to match original behavior
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        batch, channels, seqlen = x.shape
        fft_size = 2 * seqlen

        # Compute rfft using PyTorch (real input -> complex output, length = 2*seqlen)
        # Output shape: (batch, channels, seqlen+1)
        x_f32 = x  # already float32
        x_freq = torch.fft.rfft(x_f32, n=fft_size)

        # Extract real and imaginary parts (float32, contiguous)
        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        # Allocate outputs
        out_real = torch.empty_like(x_freq_real)
        out_imag = torch.empty_like(x_freq_imag)

        # Total number of elements
        N_total = x_freq_real.numel()  # batch * channels * (seqlen+1)

        # Launch Triton kernel to normalize
        BLOCK = 1024  # tile size; 1024 is a good default for simple elementwise ops
        grid = (triton.cdiv(N_total, BLOCK),)
        norm = float(2 * seqlen)  # original code divides by 2*seqlen

        rfft_divide_kernel[grid](
            x_freq_real, x_freq_imag,
            out_real, out_imag,
            N_total, norm,
            BLOCK=BLOCK,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
