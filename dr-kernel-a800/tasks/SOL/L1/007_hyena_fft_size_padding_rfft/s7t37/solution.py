import torch

# Triton availability guard (environment may not have Triton)
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: zero-pad to N_in = 2 * seqlen, compute real DFT, normalize, write first L_out = N_in//2 + 1 outputs.
# One program per output index k.
if TRITON_AVAILABLE:
    @triton.jit
    def _rfft_zero_pad_direct_kernel(
        x_ptr,          # *const float32, flattened input of length N_in
        out_real_ptr,   # *float32, flattened output real of length L_out
        out_imag_ptr,   # *float32, flattened output imag of length L_out
        N_in: tl.constexpr,   # int: 2 * seqlen (padded input length)
        L_out: tl.constexpr,  # int: N_in // 2 + 1 (first L_out outputs, equals seqlen + 1 for N_in = 2*seqlen)
        scale: tl.constexpr,  # float: 1.0 / (2.0 * seqlen)
    ):
        # Each program computes one output y[k] for k in [0, L_out)
        k = tl.program_id(0)

        # Accumulators for real and imaginary parts
        y_real = 0.0
        y_imag = 0.0

        # Loop over all n in [0, N_in); N_in is constexpr for specialization
        for n in range(0, N_in):
            # Load x[n] as float32
            x_n = tl.load(x_ptr + n)

            # Compute angle = -2 * pi * k * n / N_in
            angle = -(2.0 * tl.pi) * (k * n) / N_in

            # Real and imaginary contributions
            contrib_real = x_n * tl.cos(angle)
            contrib_imag = x_n * tl.sin(angle)

            # Accumulate
            y_real += contrib_real
            y_imag += contrib_imag

        # Normalize by scale
        y_real = y_real * scale
        y_imag = y_imag * scale

        # Store results for k-th output (0 <= k < L_out)
        tl.store(out_real_ptr + k, y_real)
        tl.store(out_imag_ptr + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input x: (batch, channels, seqlen), float32 tensor.
        - Compute zero-padded real FFT with n = 2*seqlen, output length seqlen + 1.
        - Normalize by 2*seqlen.
        - Return real and imaginary parts, each of shape (batch, channels, seqlen + 1).
        """
        # Ensure Triton is available; if not, raise as the evaluation requires Triton execution.
        if not TRITON_AVAILABLE:
            raise RuntimeError("ModelNew requires Triton.")

        # Extract shape
        batch, channels, seqlen = x.shape

        # Padded input length and output length
        N_in = 2 * seqlen
        L_out = N_in // 2 + 1  # equals seqlen + 1

        # Allocate outputs: real and imaginary parts (flattened for Triton)
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_real_flat = out_real.view(-1)
        out_imag_flat = out_imag.view(-1)

        # Normalization scale
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernel: one program per output index k in [0, L_out)
        grid = (L_out,)

        # Pass N_in and L_out as meta-parameters (constexpr for Triton)
        _rfft_zero_pad_direct_kernel[grid](
            x.view(-1),       # pass flattened input (expects float32)
            out_real_flat,    # flattened output real
            out_imag_flat,    # flattened output imag
            N_in=N_in,
            L_out=L_out,
            scale=scale,
        )

        # Return real and imaginary parts; shape matches original (batch, channels, seqlen + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
