import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr, seqlen, n_elements, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..n_elements-1} x[k] * cos(2*pi*j*k/n_elements) / n_elements
      for j in 0..seqlen.
    x_ptr: pointer to a 1D vector (length n_elements == 2*seqlen).
    out_ptr: pointer to a 1D output vector (we write indices j=0..seqlen).
    """
    pid = tl.program_id(axis=0)  # one program per (batch, channel) row
    for j in range(0, seqlen + 1):
        acc = 0.0
        for k in range(0, n_elements, BLOCK_K):
            offs = k + tl.arange(0, BLOCK_K)
            mask = offs < n_elements
            xk = tl.load(x_ptr + offs, mask=mask, other=0.0)
            angle = 2.0 * 3.141592653589793 * j * offs / n_elements
            ck = tl.cos(angle)
            ck = tl.where(mask, ck, 0.0)
            acc += tl.sum(xk * ck, axis=0)
        val = acc / n_elements
        out_off = pid * (seqlen + 1) + j
        tl.store(out_ptr + out_off, val)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr, seqlen, n_elements, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..n_elements-1} x[k] * sin(2*pi*j*k/n_elements) / n_elements
      for j in 1..seqlen-1.
    x_ptr: pointer to a 1D vector (length n_elements == 2*seqlen).
    out_ptr: pointer to a 1D output vector (we write indices j=1..seqlen-1).
    """
    pid = tl.program_id(axis=0)  # one program per (batch, channel) row
    for j in range(1, seqlen):
        acc = 0.0
        for k in range(0, n_elements, BLOCK_K):
            offs = k + tl.arange(0, BLOCK_K)
            mask = offs < n_elements
            xk = tl.load(x_ptr + offs, mask=mask, other=0.0)
            angle = 2.0 * 3.141592653589793 * j * offs / n_elements
            sk = tl.sin(angle)
            sk = tl.where(mask, sk, 0.0)
            acc += tl.sum(xk * sk, axis=0)
        val = acc / n_elements
        out_off = pid * (seqlen + 1) + j
        tl.store(out_ptr + out_off, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: input tensor of shape (batch, channels, seqlen), float32, on CUDA.
        Returns:
          x_freq_real: float32 tensor of shape (batch, channels, seqlen+1)
          x_freq_imag: float32 tensor of shape (batch, channels, seqlen+1)
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        n_elements = 2 * seqlen

        # Allocate outputs; imag_out[0] and imag_out[seqlen] will be set to zero.
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag[:, :, 0] = 0.0
        x_freq_imag[:, :, seqlen] = 0.0

        # One program per (batch, channel) row
        grid = (batch * channels,)

        # Launch real part kernel: x_ptr points to 1D data of length n_elements.
        # The forward path only allocates and launches; no torch computation.
        rfft_real_kernel[grid](x, x_freq_real, seqlen, n_elements, BLOCK_K=1024, num_warps=4)

        # Launch imaginary part kernel for j in 1..seqlen-1
        rfft_imag_kernel[grid](x, x_freq_imag, seqlen, n_elements, BLOCK_K=1024, num_warps=4)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
