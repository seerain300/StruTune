import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      seqlen, N):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N
      for j in 0..seqlen, where N = 2 * seqlen.
    x_ptr points to the padded input vector (length N), per-row base.
    out_ptr points to output vector of length (seqlen + 1), per-row base.
    """
    pid = tl.program_id(axis=0)
    # j loop: j = 0..seqlen
    for j in range(0, seqlen + 1):
        acc = 0.0
        # k loop: k = 0..N-1
        for k in range(0, N):
            xk = tl.load(x_ptr + k)
            angle = tl.float32(2.0 * 3.141592653589793 * j * k / N)
            cosk = tl.cos(angle)
            contrib = xk * cosk
            acc += contrib
        # Normalize by 2*seqlen (i.e., N)
        acc = acc / N
        # Store to output at index j
        tl.store(out_ptr + pid * (seqlen + 1) + j, acc)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      seqlen, N):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N
      for j in 1..seqlen-1, where N = 2 * seqlen.
      imag_out[0] = 0, imag_out[seqlen] = 0.
    x_ptr points to the padded input vector (length N), per-row base.
    out_ptr points to output vector of length (seqlen + 1), per-row base.
    """
    pid = tl.program_id(axis=0)
    # j loop: j = 1..seqlen-1
    for j in range(1, seqlen):
        acc = 0.0
        # k loop: k = 0..N-1
        for k in range(0, N):
            xk = tl.load(x_ptr + k)
            angle = tl.float32(2.0 * 3.141592653589793 * j * k / N)
            sink = tl.sin(angle)
            contrib = xk * sink
            acc += contrib
        acc = acc / N
        tl.store(out_ptr + pid * (seqlen + 1) + j, acc)
    # Set imag_out[0] and imag_out[seqlen] to zero (they must be zero for real rfft)
    tl.store(out_ptr + pid * (seqlen + 1) + 0, 0.0)
    tl.store(out_ptr + pid * (seqlen + 1) + seqlen, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused FFT size padding and real FFT computation for Hyena convolution.
        Returns:
            x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1)
            x_freq_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1)
        """
        # Expect x of shape (batch, channels, seqlen)
        assert x.ndim == 3, "Input must be 3D: (batch, channels, seqlen)"
        # Ensure dtype float32 for computation
        x = x.to(torch.float32)
        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Flatten rows for easier processing: rows = batch * channels
        rows = batch * channels
        x_flat = x.contiguous().view(rows, seqlen)

        # Allocate padded input per row: length N, float32
        padded_x = torch.empty((rows, N), dtype=torch.float32, device=x.device)
        # Copy the first seqlen elements
        padded_x[:, :seqlen] = x_flat
        # Fill the rest with zeros
        if N > seqlen:
            padded_x[:, seqlen:] = 0.0

        # Allocate outputs: (rows, seqlen+1) float32
        out_real = torch.empty((rows, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((rows, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels: one program per row
        grid = (rows,)
        rfft_real_kernel[grid](padded_x, out_real, seqlen, N, num_warps=1)
        rfft_imag_kernel[grid](padded_x, out_imag, seqlen, N, num_warps=1)

        # Reshape back to (batch, channels, seqlen+1)
        x_freq_real = out_real.view(batch, channels, seqlen + 1)
        x_freq_imag = out_imag.view(batch, channels, seqlen + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
