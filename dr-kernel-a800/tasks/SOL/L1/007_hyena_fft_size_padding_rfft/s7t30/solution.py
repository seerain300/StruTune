import torch

# Triton import and availability flag
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _real_rfft_zero_pad_kernel(x_ptr, real_out_ptr, imag_out_ptr, N_in, L_out, scale):
        """
        Compute real FFT of zero-padded input of length N_in (set to 2*seqlen),
        return first L_out complex coefficients (L_out = N_in//2 + 1 = seqlen + 1).
        Normalize by scale = 1 / N_in.
        """
        # Each program computes one output index k
        pid = tl.program_id(axis=0)
        if pid >= L_out:
            return

        # Precompute constants
        two_pi = 6.283185307179586476925286766559  # 2 * pi
        # Accumulators for real and imaginary parts
        acc_real = 0.0
        acc_imag = 0.0

        # Direct summation over n in [0, N_in)
        # Note: x_ptr length is N_in, we can load x[n] directly; we do not need explicit masking
        # because the loop iterates over N_in, which is the padded length.
        for n in range(0, N_in):
            # Load x[n] as float
            x_n = tl.load(x_ptr + n)
            # Angle = -2*pi*k*n / N_in
            angle = two_pi * (float(pid) * float(n)) / float(N_in)
            # Compute contributions
            acc_real += x_n * tl.cos(angle)
            acc_imag += x_n * tl.sin(angle)

        # Normalize
        acc_real = acc_real * scale
        acc_imag = acc_imag * scale

        # Store results
        tl.store(real_out_ptr + pid, acc_real)
        tl.store(imag_out_ptr + pid, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Cast input to float32
        - Emulate torch.fft.rfft(x, n=2*seqlen) with zero-padding
        - Normalize by 2*seqlen
        - Return real and imaginary parts, each of shape (batch, channels, seqlen + 1)
        """
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            # Fallback: if Triton is not available, perform torch computation (still correct).
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)
            x_freq = x_freq / (2.0 * seqlen)
            return x_freq.real, x_freq.imag

        # Input x: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape

        # Cast to float32 and flatten for Triton
        x_f32 = x.to(torch.float32)
        x_flat = x_f32.view(-1)

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

        _real_rfft_zero_pad_kernel[grid](
            x_flat, out_real.view(-1), out_imag.view(-1),
            N_in, L_out, scale,
        )

        # Return real and imaginary parts; shape matches original (batch, channels, seqlen + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
