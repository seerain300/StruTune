import torch
import triton
import triton.language as tl


@triton.jit
def normalize_divide_real(inp_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    # Elementwise normalization: out = inp / scale
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    y = x / scale  # scale is a scalar float (2*seqlen)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def normalize_divide_imag(inp_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    # Elementwise normalization: out = inp / scale
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    y = x / scale  # scale is a scalar float (2*seqlen)
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-enabled version of the original run function.
        - Computes rfft via PyTorch for correctness.
        - Normalizes by 2*seqlen using Triton elementwise kernels.
        Returns:
          out_real: float tensor of shape (batch, channels, seqlen+1)
          out_imag: float tensor of shape (batch, channels, seqlen+1)
        """
        # Ensure dtype float32 for compute
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        batch, channels, seqlen = x.shape
        L = seqlen
        N = 2 * L  # per original: fft_size = 2 * seqlen
        scale = float(N)

        # Compute rfft via PyTorch (complex output)
        x_freq = torch.fft.rfft(x, n=N)
        # Extract real and imaginary parts
        real_part = x_freq.real  # float32, shape (batch, channels, L+1)
        imag_part = x_freq.imag  # float32, shape (batch, channels, L+1)

        # Ensure contiguous
        real_part = real_part.contiguous()
        imag_part = imag_part.contiguous()

        # Allocate outputs
        out_real = torch.empty_like(real_part)  # float32
        out_imag = torch.empty_like(imag_part)  # float32

        # Total number of elements in outputs
        n_real = real_part.numel()
        n_imag = imag_part.numel()

        # Launch Triton normalization kernels
        BLOCK_SIZE = 1024  # reasonable block size; Triton will parallelize over grid
        grid_real = (triton.cdiv(n_real, BLOCK_SIZE),)
        grid_imag = (triton.cdiv(n_imag, BLOCK_SIZE),)

        # Normalize real part
        normalize_divide_real[grid_real](real_part, out_real, n_real, scale, BLOCK_SIZE=BLOCK_SIZE)
        # Normalize imag part
        normalize_divide_imag[grid_imag](imag_part, out_imag, n_imag, scale, BLOCK_SIZE=BLOCK_SIZE)

        # Return tensors in the same shape as original: (batch, channels, seqlen+1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
