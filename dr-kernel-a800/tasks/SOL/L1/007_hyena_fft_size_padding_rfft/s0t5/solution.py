import torch
import triton
import triton.language as tl


@triton.jit
def normalize_divide_kernel(in_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: elementwise divide each element in 'in_ptr' by 'scale' and write to 'out_ptr'.
    Assumes float32 tensors. Flattened 1D indexing.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    # scale is a Python float; Triton handles division with float input
    y = x / scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute rfft using PyTorch (as per original), then normalize real and imaginary parts
        using Triton kernels. Returns (batch, channels, seqlen+1) for real and imaginary.
        """
        # Ensure float32 for numerical stability (original code casts to float32)
        x_f32 = x.to(torch.float32)

        batch, channels, seqlen = x_f32.shape
        N = 2 * seqlen

        # Compute rfft with padding to N (real input -> complex output)
        # Note: Using torch here ensures correctness; Triton handles normalization.
        x_freq = torch.fft.rfft(x_f32, n=N)  # shape: (batch, channels, seqlen+1)

        # Extract real and imaginary parts
        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        # Normalize by N (2*seqlen)
        n_real = x_freq_real.numel()
        n_imag = x_freq_imag.numel()
        scale = float(N)  # 2 * seqlen as float for Triton

        # Launch Triton kernels for normalization
        BLOCK_SIZE = 1024
        grid_real = (triton.cdiv(n_real, BLOCK_SIZE),)
        grid_imag = (triton.cdiv(n_imag, BLOCK_SIZE),)

        # Output tensors for normalized real/imag
        out_real = torch.empty_like(x_freq_real)
        out_imag = torch.empty_like(x_freq_imag)

        # Run kernels
        normalize_divide_kernel[grid_real](x_freq_real, out_real, n_real, scale, BLOCK_SIZE=BLOCK_SIZE)
        normalize_divide_kernel[grid_imag](x_freq_imag, out_imag, n_imag, scale, BLOCK_SIZE=BLOCK_SIZE)

        # Return real and imaginary parts (already matching shape (batch, channels, seqlen+1))
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
