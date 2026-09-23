import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_bins_kernel(x_ptr, out_real_ptr, N, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft bins for a single (batch, channel) row:
      out_real[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N, for j in 0..seqlen.
    x_ptr: float32 pointer to input vector of length at least seqlen. We emulate zero-padding
           by masked loads where idx >= 2*seqlen.
    out_real_ptr: float32 pointer to output real vector (we write j in 0..seqlen).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    j = tl.program_id(2)  # grid dimension for bins, j in 0..seqlen

    acc = 0.0

    # Iterate over k in chunks of BLOCK_K
    for k in range(0, N, BLOCK_K):
        offs = tl.arange(0, BLOCK_K)
        idx = k + offs
        mask = idx < N  # true for idx < 2*seqlen

        # Emulate zero-padding by masked loads beyond 2*seqlen
        x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)

        theta = 2.0 * 3.141592653589793 * (j * idx / N)
        cos_vals = tl.cos(theta)

        # Reduce across the chunk
        acc += tl.sum(x_vals * cos_vals, axis=0)

    acc = acc / N
    tl.store(out_real_ptr + j, acc)


@triton.jit
def rfft_imag_bins_kernel(x_ptr, out_imag_ptr, N, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft bins for a single (batch, channel) row:
      out_imag[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N, for j in 1..seqlen-1.
    x_ptr: float32 pointer to input vector of length at least seqlen. We emulate zero-padding
           by masked loads where idx >= 2*seqlen.
    out_imag_ptr: float32 pointer to output imag vector (we write j in 1..seqlen-1).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    j = tl.program_id(2)  # grid dimension for bins, j in 1..seqlen-1

    acc = 0.0

    for k in range(0, N, BLOCK_K):
        offs = tl.arange(0, BLOCK_K)
        idx = k + offs
        mask = idx < N  # true for idx < 2*seqlen

        x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)

        theta = 2.0 * 3.141592653589793 * (j * idx / N)
        sin_vals = tl.sin(theta)

        acc += tl.sum(x_vals * sin_vals, axis=0)

    acc = acc / N
    tl.store(out_imag_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation: compute rfft(x, n=2*seqlen), normalize by 2*seqlen,
        and return real and imaginary parts separately as float32 tensors of shape
        (batch, channels, seqlen+1). No torch operations in forward; Triton kernels are launched.
        """
        # Ensure input is float32
        assert x.dtype == torch.float32, "Input must be float32."

        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Allocate outputs (device-side)
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch real part: j in 0..seqlen (total seqlen+1 bins)
        grid_real = (batch, channels, seqlen + 1)
        rfft_real_bins_kernel[grid_real](x, out_real, N, seqlen, BLOCK_K=1024, num_warps=4)

        # Launch imag part: j in 1..seqlen-1 (seqlen-1 bins); imag_out[0] and imag_out[seqlen] are zero.
        grid_imag = (batch, channels, seqlen - 1)
        rfft_imag_bins_kernel[grid_imag](x, out_imag, N, seqlen, BLOCK_K=1024, num_warps=4)
        # Set imag_out[0] and imag_out[seqlen] to zero
        out_imag[:, :, 0] = 0.0
        out_imag[:, :, seqlen] = 0.0

        # Return real and imaginary parts
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
