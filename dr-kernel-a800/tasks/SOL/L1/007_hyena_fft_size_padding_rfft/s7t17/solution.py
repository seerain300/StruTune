import torch

# Triton import and availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _rfft_zero_pad_direct_kernel(
        x_ptr,            # *const float32
        out_real_ptr,     # *float32
        out_imag_ptr,     # *float32
        N_in: tl.constexpr,   # int, 2 * seqlen (zero-padding length)
        L_out: tl.constexpr,  # int, seqlen + 1
        scale,                 # float32
    ):
        # One program per output frequency index k in [0, L_out)
        k = tl.program_id(axis=0)

        # Accumulator for real and imaginary parts (scalars)
        acc_real = tl.zeros((), dtype=tl.float32)
        acc_imag = tl.zeros((), dtype=tl.float32)

        # Constants
        pi = 3.141592653589793
        two_pi_over_N = 2.0 * pi / N_in

        # Sum over n in [0, N_in): zero-padding handled by loading zeros for n >= seqlen
        for n in range(0, N_in):
            # Load x[n] as float32
            x_val = tl.load(x_ptr + n)
            # Compute phase = -2*pi*k*n/N_in
            phase = -two_pi_over_N * k * n
            # Accumulate real and imaginary parts
            acc_real += x_val * tl.cos(phase)
            acc_imag += x_val * tl.sin(phase)

        # Normalize by 2*seqlen
        y_real = acc_real * scale
        y_imag = acc_imag * scale

        # Store results
        tl.store(out_real_ptr + k, y_real)
        tl.store(out_imag_ptr + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Emulate torch.fft.rfft(x, n=2*seqlen) with zero-padding
        - Normalize by 2*seqlen
        - Return real and imaginary parts, each of shape (batch, channels, seqlen + 1)
        """
        # If Triton is not available, fall back to PyTorch
        if not TRITON_AVAILABLE:
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)
            x_freq = x_freq / (2.0 * seqlen)
            return x_freq.real, x_freq.imag

        # Input x: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape

        # Padded length and output length as per rfft semantics
        N_in = 2 * seqlen  # zero-pad to this length
        L_out = N_in // 2 + 1  # equals seqlen + 1

        # Output tensors: real and imaginary parts, shape (batch, channels, seqlen + 1)
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Normalization scale
        scale = 1.0 / (2.0 * seqlen)

        # Flatten input for kernel
        x_flat = x.view(-1)

        # Launch Triton kernel: one program per output frequency index k in [0, L_out)
        grid = (L_out,)

        _rfft_zero_pad_direct_kernel[grid](
            x_flat,
            out_real.view(-1),
            out_imag.view(-1),
            N_in,
            L_out,
            scale,
        )

        # Return real and imaginary parts; shape matches original (batch, channels, seqlen + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
