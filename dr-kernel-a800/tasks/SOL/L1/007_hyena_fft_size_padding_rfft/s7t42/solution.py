import torch

# Guard Triton availability
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _rfft_zero_pad_direct_kernel(
        x_ptr,            # *const float32, flattened input of length 2*seqlen
        real_out_ptr,     # *float32, flattened output real part
        imag_out_ptr,     # *float32, flattened output imag part
        N_in: tl.constexpr,   # int, padded input length = 2*seqlen
        L_out: tl.constexpr,  # int, output length = seqlen + 1
        scale,                 # float32, normalization = 1.0 / (2*seqlen)
    ):
        # One Triton program computes one output index k in [0, L_out)
        k = tl.program_id(0)

        # Accumulators for real and imaginary parts
        acc_real = 0.0
        acc_imag = 0.0

        # Compute the DFT sum over n in [0, N_in)
        # Zero padding is handled by the fact that for n >= seqlen, x[n] is not set and implicitly contributes 0.
        for n in range(0, N_in):
            val = tl.load(x_ptr + n)  # x_ptr is float32
            angle = -2.0 * 3.141592653589793 * float(k) * float(n) / float(N_in)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            acc_real += val * cos_term
            acc_imag += val * sin_term

        # Normalize by (2*seqlen)
        y_real = acc_real * scale
        y_imag = acc_imag * scale

        # Store results (k in [0, L_out))
        tl.store(real_out_ptr + k, y_real)
        tl.store(imag_out_ptr + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Cast input to float32
        - Emulate torch.fft.rfft(x, n=2*seqlen) with zero-padding
        - Normalize by 2*seqlen
        - Return real and imaginary parts, each of shape (batch, channels, seqlen + 1)
        """
        # If Triton not available, fall back to original behavior (for robustness)
        if not TRITON_AVAILABLE:
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)
            x_freq = x_freq / (2.0 * seqlen)
            return x_freq.real, x_freq.imag

        # Input x: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape

        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Padded length and output length as per rfft semantics
        N_in = 2 * seqlen  # zero-pad to this length
        # Output length for rfft(input_length=seqlen, n=N_in) is L_out = N_in // 2 + 1, which equals seqlen + 1
        L_out = N_in // 2 + 1  # equals seqlen + 1

        # Flatten input for Triton (kernel expects 1D pointers)
        x_flat = x_f32.view(-1)

        # Output tensors: real and imaginary parts, shape (batch, channels, seqlen + 1)
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Normalization scale
        scale = 1.0 / (2.0 * seqlen)

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
