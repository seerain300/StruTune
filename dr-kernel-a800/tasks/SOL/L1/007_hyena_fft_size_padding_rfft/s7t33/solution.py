import torch

# Triton import guarded; ensure the environment provides Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _rfft_zero_pad_kernel(
        x_ptr,                 # *const float, flattened input (length = 2*seqlen)
        out_real_ptr,          # *float, flattened output real (length = seqlen + 1)
        out_imag_ptr,          # *float, flattened output imag (length = seqlen + 1)
        N_in,                  # int: padded input length = 2*seqlen
        L_out,                 # int: output length = seqlen + 1
        scale,                 # float32: 1.0 / (2.0 * seqlen)
    ):
        # One program per output index k in [0, L_out)
        k = tl.program_id(0)
        # Accumulators for real and imag parts
        sum_real = 0.0
        sum_imag = 0.0

        # Direct DFT over zero-padded input of length N_in
        # y_real[k] = sum_{n=0}^{N_in-1} x[n] * cos(2*pi*k*n / N_in)
        # y_imag[k] = -sum_{n=0}^{N_in-1} x[n] * sin(2*pi*k*n / N_in)
        for n in range(N_in):
            # Load x[n]; zero-padding is handled by indexing outside original seqlen range.
            # Note: Triton will allow arithmetic with int; ensure N_in is int.
            x_val = tl.load(x_ptr + n)
            angle = 2.0 * 3.141592653589793 * float(k) * float(n) / float(N_in)
            # Accumulate cosine and sine contributions
            sum_real += x_val * tl.cos(angle)
            sum_imag += x_val * tl.sin(angle)

        # Normalize by 2*seqlen
        y_real = sum_real * scale
        y_imag = sum_imag * scale

        # Store results
        tl.store(out_real_ptr + k, y_real)
        tl.store(out_imag_ptr + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input x: (batch, channels, seqlen) float32.
        - Emulate torch.fft.rfft(x, n=2*seqlen) with zero-padding and return normalized real/imag parts.
        - Output: (batch, channels, seqlen + 1) for both real and imag.
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for ModelNew forward.")

        # Input shape
        batch, channels, seqlen = x.shape

        # Padded length and output length (as per rfft semantics)
        N_in = 2 * seqlen  # zero-pad to this length
        L_out = N_in // 2 + 1  # equals seqlen + 1

        # Output tensors: real and imaginary parts, shape (batch, channels, seqlen + 1)
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Normalization scale
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernel: one program per output frequency index k in [0, L_out)
        grid = (L_out,)

        # Flatten input and ensure float32
        x_flat = x.contiguous().view(-1).to(torch.float32)

        _rfft_zero_pad_kernel[grid](
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
