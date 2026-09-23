import torch

# Triton import and availability guard
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute zero-padded real FFT of length N_in and return the first L_out bins (real and imag),
# normalized by N_in. Launch with one program per output index k.
@triton.jit
def _rfft_zero_pad_direct_kernel(
    x_ptr,           # *const float, pointer to input flattened
    out_real_ptr,    # *float, pointer to output real flattened
    out_imag_ptr,    # *float, pointer to output imag flattened
    N_in: tl.constexpr,     # int, padded input length (2 * seqlen)
    L_out: tl.constexpr,    # int, output length (N_in // 2 + 1) == seqlen + 1
    scale: tl.constexpr,    # float, normalization factor (1.0 / (2.0 * seqlen))
):
    # Each program computes one output index k (in [0, L_out))
    k = tl.program_id(0)

    # Accumulators in float32
    acc_real = tl.zeros((), dtype=tl.float32)
    acc_imag = tl.zeros((), dtype=tl.float32)

    # Sum over n from 0 to N_in - 1
    for n in range(N_in):
        # Load input; x_ptr points to flattened float tensor
        x_val = tl.load(x_ptr + n)

        # Compute angle: -2*pi*k*n/N_in
        angle = -(2.0 * 3.141592653589793) * k * n / N_in

        # Complex exponential: cos(angle) + i*sin(angle)
        x_f = tl.cast(x_val, tl.float32)
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)

        acc_real += x_f * cos_term
        acc_imag += x_f * sin_term

    # Normalize by N_in (2*seqlen)
    y_real = acc_real * scale
    y_imag = acc_imag * scale

    # Store results
    tl.store(out_real_ptr + k, y_real)
    tl.store(out_imag_ptr + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Treat input as real, zero-pad to 2*seqlen
        - Compute real FFT and normalize by 2*seqlen
        - Return real and imaginary parts, each of shape (batch, channels, seqlen + 1)
        """
        # Ensure Triton is available; if not, raise to avoid silent incorrect behavior.
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required but not available.")

        # x is (batch, channels, seqlen); do not use torch ops here.
        batch, channels, seqlen = x.shape

        # Padded length and output length
        N_in = 2 * seqlen  # zero-pad to this length
        L_out = N_in // 2 + 1  # equals seqlen + 1

        # Flatten input pointer (assume x is float32 as in the original usage).
        # If x is not float32, evaluation typically converts it; here we treat it as provided float32.
        x_flat = x.view(-1)

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
