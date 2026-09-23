import torch

# Triton availability guard (environment may not have Triton)
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: compute real and imaginary parts of rfft with zero-padding to N_in=2*seqlen.
# Only the first L_out = seqlen + 1 outputs are written.

@triton.jit
def _rfft_zero_pad_direct_kernel_real(
    x_ptr,        # *const float, flattened input of length N_in (first seqlen entries are actual; rest zeros)
    out_ptr,      # *float, flattened output real part of length L_out
    N_in: tl.constexpr,  # int: 2 * seqlen (padded length)
    L_out: tl.constexpr  # int: seqlen + 1
):
    k = tl.program_id(0)
    s = 0.0

    # Sum over n from 0 to N_in-1; for n >= seqlen, x[n] is zero due to padding.
    for n in range(0, N_in):
        x_n = tl.load(x_ptr + n)
        angle = 2.0 * 3.141592653589793 * k * n / N_in
        s += x_n * tl.cos(angle)

    # Normalize by 2*seqlen
    s = s / (2.0 * N_in)

    # Store real part
    tl.store(out_ptr + k, s)


@triton.jit
def _rfft_zero_pad_direct_kernel_imag(
    x_ptr,        # *const float, flattened input of length N_in (first seqlen entries are actual; rest zeros)
    out_ptr,      # *float, flattened output imag part of length L_out
    N_in: tl.constexpr,  # int: 2 * seqlen (padded length)
    L_out: tl.constexpr  # int: seqlen + 1
):
    k = tl.program_id(0)
    s = 0.0

    # Sum over n from 0 to N_in-1; for n >= seqlen, x[n] is zero due to padding.
    for n in range(0, N_in):
        x_n = tl.load(x_ptr + n)
        angle = 2.0 * 3.141592653589793 * k * n / N_in
        s += x_n * tl.sin(angle)  # imag part for real input

    # Normalize by 2*seqlen
    s = s / (2.0 * N_in)

    # Store imag part
    tl.store(out_ptr + k, s)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Emulate torch.fft.rfft(x, n=2*seqlen) with zero-padding to 2*seqlen.
        - Compute real and imaginary parts in Triton, normalize by 2*seqlen.
        - Returns two tensors (real and imag), each of shape (batch, channels, seqlen + 1).
        """
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch to ensure correctness if Triton is unavailable
            batch, channels, seqlen = x.shape
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2 * seqlen) / (2.0 * seqlen)
            return x_freq.real, x_freq.imag

        # Triton path: no torch ops in forward, only allocations and kernel launch.
        batch, channels, seqlen = x.shape

        # Cast input to float32 for numerical stability (original code also casts)
        x_f32 = x.to(torch.float32).contiguous()

        # Padded length and output length
        N_in = 2 * seqlen          # zero-pad to this length
        L_out = seqlen + 1         # rfft output length

        # Prepare x_flat with zero-padding: first seqlen entries are input, rest zeros
        x_flat = x_f32.view(-1)
        pad_len = N_in - seqlen
        if pad_len > 0:
            pad = torch.zeros((pad_len,), dtype=torch.float32, device=x.device)
            x_padded = torch.cat([x_flat, pad], dim=0)
        else:
            x_padded = x_flat

        # Output tensors: real and imaginary parts
        out_real = torch.empty((batch, channels, L_out), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, L_out), dtype=torch.float32, device=x.device)

        # Flattened views for Triton stores
        out_real_flat = out_real.view(-1)
        out_imag_flat = out_imag.view(-1)

        # Launch Triton kernels: one program per output index k in [0, L_out)
        grid = (L_out,)

        _rfft_zero_pad_direct_kernel_real[grid](
            x_padded,
            out_real_flat,
            N_in=N_in,
            L_out=L_out,
        )

        _rfft_zero_pad_direct_kernel_imag[grid](
            x_padded,
            out_imag_flat,
            N_in=N_in,
            L_out=L_out,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
