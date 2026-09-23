import torch
import triton
import triton.language as tl


@triton.jit
def cosine_direct_kernel(x_ptr, out_real_ptr, seqlen: tl.constexpr):
    """
    For each j in [0, seqlen], compute:
      real[j] = sum_{k=0..seqlen-1} x[k] * (cos(2*pi*j*k/(2*seqlen)) + (-1)^k * cos(2*pi*j*(seqlen - k)/(2*seqlen))) / (2*seqlen)
    and write to out_real_ptr[row * (seqlen+1) + j].
    x_ptr length is seqlen, row is the flattened (batch, channel) index.
    """
    # We will loop over j and k; Triton supports dynamic loops over runtime values.
    # Note: Triton prefers compile-time constants for tl.range, but seqlen is passed as tl.constexpr via argument.
    # However, Triton handles runtime integers as well for Python-side loops.
    for j in range(0, seqlen + 1):
        acc = 0.0
        for k in range(0, seqlen):
            xk = tl.load(x_ptr + k)  # x_ptr points to the start of the row
            # Compute angles
            angle1 = 2.0 * 3.141592653589793 * j * k / (2.0 * seqlen)
            angle2 = 2.0 * 3.141592653589793 * j * (seqlen - k) / (2.0 * seqlen)
            parity = 1.0 if (k % 2 == 0) else -1.0
            contrib = xk * (tl.cos(angle1) + parity * tl.cos(angle2))
            acc += contrib
        # Normalize by 2*seqlen
        acc *= 1.0 / (2.0 * seqlen)
        # Store to out_real[row * (seqlen+1) + j]
        tl.store(out_real_ptr + (j), acc)


@triton.jit
def sine_direct_kernel(x_ptr, out_imag_ptr, seqlen: tl.constexpr):
    """
    For each j in [1, seqlen-1], compute:
      imag[j] = sum_{k=0..seqlen-1} x[k] * (sin(2*pi*j*k/(2*seqlen)) - (-1)^k * sin(2*pi*j*(seqlen - k)/(2*seqlen))) / (2*seqlen)
    and write to out_imag_ptr[row * (seqlen+1) + j].
    x_ptr length is seqlen, row is the flattened (batch, channel) index.
    """
    for j in range(1, seqlen):
        acc = 0.0
        for k in range(0, seqlen):
            xk = tl.load(x_ptr + k)
            angle1 = 2.0 * 3.141592653589793 * j * k / (2.0 * seqlen)
            angle2 = 2.0 * 3.141592653589793 * j * (seqlen - k) / (2.0 * seqlen)
            parity = 1.0 if (k % 2 == 0) else -1.0
            contrib = xk * (tl.sin(angle1) - parity * tl.sin(angle2))
            acc += contrib
        # Normalize by 2*seqlen
        acc *= 1.0 / (2.0 * seqlen)
        tl.store(out_imag_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          x_f32 = x.to(torch.float32)
          x_freq = torch.fft.rfft(x_f32, n=2*seqlen)  # complex of shape (B, C, seqlen+1)
          x_freq = x_freq / (2*seqlen)
          return real part and imag part separately as float32 tensors of shape (B, C, seqlen+1)
        We avoid torch.fft in forward and compute via Triton kernels.
        """
        # Ensure input is float32 and contiguous
        x_f32 = x.to(torch.float32).contiguous()

        batch, channels, seqlen = x_f32.shape
        N2 = 2 * seqlen  # even, per original code
        BC = batch * channels

        # Flatten (batch, channels) into rows
        x_flat = x_f32.view(BC, seqlen)

        # Allocate outputs (real and imag) as float32, length (BC, seqlen+1)
        real_out = torch.empty((BC, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((BC, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels for each row
        # cosine_direct_kernel computes real part for j in 0..seqlen
        cosine_direct_kernel[(BC,)](x_flat, real_out, seqlen=seqlen)

        # sine_direct_kernel computes imag part for j in 1..seqlen-1
        sine_direct_kernel[(BC,)](x_flat, imag_out, seqlen=seqlen)

        # Set imag[0] and imag[seqlen] to zero (exact for real inputs)
        imag_out[:, 0] = 0.0
        imag_out[:, seqlen] = 0.0

        # Reshape back to (batch, channels, seqlen+1)
        real_out = real_out.view(batch, channels, seqlen + 1)
        imag_out = imag_out.view(batch, channels, seqlen + 1)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
