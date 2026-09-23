import triton
import triton.language as tl

# Triton kernel: compute normalized real FFT outputs for k in [0, seqlen+1) only.
# We treat input length as L = seqlen and compute:
# y_real[k] = sum_{n=0..L-1} x[n] * cos(2*pi*k*n/(2*L))
# y_imag[k] = sum_{n=0..L-1} x[n] * sin(2*pi*k*n/(2*L))
# Then divide by 2*L. This matches torch.fft.rfft(x, n=2*L)[:seqlen+1].real/imag for real x.
@triton.jit
def _real_rfft_k_only_kernel(x_ptr, real_out_ptr, imag_out_ptr, seqlen, scale):
    k = tl.program_id(axis=0)
    # Accumulators
    acc_real = 0.0
    acc_imag = 0.0

    L = seqlen
    N_in = 2 * L  # padding length

    # Loop over n = 0..L-1 (no zero-padding needed since we compute only the first L+1 outputs)
    # For k in [0, L+1), the contribution only depends on n < L for real inputs.
    for n in range(0, L):
        x_n = tl.load(x_ptr + n)
        angle = 2.0 * tl.math.pi * k * n / N_in
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)
        acc_real += x_n * cos_term
        acc_imag += x_n * sin_term

    # Normalize
    acc_real = acc_real * scale
    acc_imag = acc_imag * scale

    # Store
    tl.store(real_out_ptr + k, acc_real)
    tl.store(imag_out_ptr + k, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input: x of shape (batch, channels, seqlen), float32.
        - Compute normalized real FFT outputs, returning real and imaginary parts,
          each of shape (batch, channels, seqlen + 1).
        """
        # Expect 3D input
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Flatten input to 1D for kernel
        x_flat = x_f32.view(-1)

        # Output tensors
        device = x.device
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=device)

        # Normalization scale: divide by 2*seqlen
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernel: one program per output index k in [0, seqlen + 1)
        grid = (seqlen + 1,)
        _real_rfft_k_only_kernel[grid](
            x_flat, out_real.view(-1), out_imag.view(-1),
            seqlen, scale,
        )

        return out_real, out_imag


# Import guard: if Triton is not available, raise ImportError to prevent any torch fallback.
try:
    TRITON_AVAILABLE = hasattr(triton, "__version__")
except Exception:
    TRITON_AVAILABLE = False

if not TRITON_AVAILABLE:
    raise ImportError("Triton is required for ModelNew; Triton not available.")


def run(*args):
    return ModelNew()(*args)
