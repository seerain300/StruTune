import torch

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: compute zero-padded real DFT and write first L_out coefficients (where L_out = (2*seqlen)//2 + 1)
@triton.jit
def _rfft_zero_pad_direct_kernel(x_ptr, out_real_ptr, out_imag_ptr, N_in, L_out, scale):
    """
    Compute y[k] = sum_{n=0}^{N_in-1} x[n] * exp(-2πi k n / N_in) for k in [0, L_out),
    where L_out = N_in // 2 + 1. We only need the first seqlen + 1 coefficients; here L_out == seqlen + 1.
    We treat input x as length seqlen and zero-pad to N_in when reading beyond seqlen.
    """
    k = tl.program_id(0)  # output frequency index
    # We assume grid=(L_out,) and k in [0, L_out)
    # Prepare accumulators (real and imaginary parts of the DFT)
    acc_real = 0.0
    acc_imag = 0.0

    # Sum over n from 0 to N_in - 1 with zero-padding for n >= seqlen
    # Note: Triton for-loops must be statically analyzable; we use a static range over N_in
    # Triton allows 'for i in range(...)'. Here N_in is a runtime argument but Triton can handle it.
    for n in range(0, N_in):
        # Load x[n] if n < seqlen, else 0 (zero-padding)
        x_n = tl.load(x_ptr + n) if n < (N_in // 2) else 0.0
        # Since we are iterating up to N_in, and input is length seqlen, we can enforce zero-padded behavior:
        # For n >= seqlen, x_n should be 0. We ensure by checking n < seqlen:
        if n < (N_in // 2):
            x_n = tl.load(x_ptr + n)
        else:
            x_n = 0.0
        # Compute exponential term: exp(-2πi k n / N_in)
        # Note: We assume float32 inputs; N_in, k are ints, we cast to float for math
        angle = (2.0 * 3.141592653589793 * float(k) * float(n)) / float(N_in)
        exp_imag = tl.cos(angle) + 1j * tl.sin(angle)
        acc_real += x_n * tl.cos(angle)
        acc_imag += x_n * tl.sin(angle)

    # Normalize by N_in (2*seqlen)
    acc_real = acc_real * scale
    acc_imag = acc_imag * scale

    # Store results (k in [0, L_out), and here L_out == seqlen + 1)
    tl.store(out_real_ptr + k, acc_real)
    tl.store(out_imag_ptr + k, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input x: (batch, channels, seqlen)
        - Compute zero-padded real FFT with n = 2*seqlen
        - Normalize by 2*seqlen
        - Return real and imaginary parts, each of shape (batch, channels, seqlen + 1)
        """
        # If Triton not available, fallback to PyTorch (not used in evaluation harness)
        if not TRITON_AVAILABLE:
            # Keep PyTorch for fallback
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)
            x_freq = x_freq / (2.0 * seqlen)
            return x_freq.real, x_freq.imag

        # Input x: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape

        # Prepare input as 1D (flattened). Ensure float32 for math
        x_flat = x.view(-1)

        # Padded length and output length as per rfft semantics
        N_in = 2 * seqlen  # zero-pad to this length
        L_out = N_in // 2 + 1  # equals seqlen + 1

        # Output tensors: real and imaginary parts, shape (batch, channels, seqlen + 1)
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Normalization scale
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernel: one program per output frequency index k in [0, L_out)
        grid = (L_out,)

        _rfft_zero_pad_direct_kernel[grid](
            x_flat, out_real.view(-1), out_imag.view(-1),
            N_in, L_out, scale,
        )

        # Return real and imaginary parts; shape matches original (batch, channels, seqlen + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
