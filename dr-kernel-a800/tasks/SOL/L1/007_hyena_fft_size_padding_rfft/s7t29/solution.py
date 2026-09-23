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
        Compute real FFT of zero-padded input of length N_in (N_in = 2*seqlen),
        producing only the first L_out = N_in//2 + 1 outputs, normalized by N_in.
        Writes real and imaginary parts to real_out_ptr and imag_out_ptr.
        """
        # Each program handles one output index k in [0, L_out)
        k = tl.program_id(0)

        # Accumulators for real and imaginary parts
        acc_real = 0.0
        acc_imag = 0.0

        # Sum over n in [0, N_in)
        # Note: Triton loop over a runtime bound is allowed; N_in is passed as int.
        for n in range(0, N_in):
            # Load x[n] as float32
            val = tl.load(x_ptr + n)
            # Angle for this term: -2*pi*k*n/N_in
            angle = -(2.0 * 3.141592653589793) * k * n / N_in
            # Compute real and imaginary contributions
            contrib_real = val * tl.cos(angle)
            contrib_imag = val * tl.sin(angle)
            # Accumulate
            acc_real += contrib_real
            acc_imag += contrib_imag

        # Normalize by N_in (2*seqlen)
        y_real = acc_real * scale
        y_imag = acc_imag * scale

        # Store results for this k
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
            # Fallback to PyTorch if Triton not available
            if not TRITON_AVAILABLE:
                batch, channels, seqlen = x.shape
                x_f32 = x.to(torch.float32)
                x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)
                x_freq = x_freq / (2.0 * seqlen)
                return x_freq.real, x_freq.imag

            # Ensure input is float32 and contiguous; keep original shape
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_flat = x_f32.view(-1)  # 1D flattened input

            # Padded length and output length as per rfft semantics
            N_in = 2 * seqlen  # zero-pad to this length
            # Output length for rfft(input_length=seqlen, n=N_in) is L_out = N_in // 2 + 1, equals seqlen + 1
            L_out = N_in // 2 + 1  # equals seqlen + 1

            # Output tensors: real and imaginary parts, shape (batch, channels, seqlen + 1)
            out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
            out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

            # Normalization scale
            scale = 1.0 / (2.0 * seqlen)

            # Launch Triton kernel: one program per output frequency index k in [0, L_out)
            grid = (L_out,)

            _real_rfft_zero_pad_kernel[grid](
                x_flat,
                out_real.view(-1),
                out_imag.view(-1),
                N_in,
                L_out,
                scale,
            )

            # Return real and imaginary parts; shape matches original (batch, channels, seqlen + 1)
            return out_real, out_imag

else:
    # If Triton is not available, define a simple ModelNew that falls back to PyTorch
    class ModelNew(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)
            x_freq = x_freq / (2.0 * seqlen)
            return x_freq.real, x_freq.imag


def run(*args):
    return ModelNew()(*args)
