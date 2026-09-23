import torch
import triton
import triton.language as tl


@triton.jit
def _real_rfft_zero_pad_kernel(x_ptr, real_out_ptr, imag_out_ptr, N, L_out, scale):
    """
    Triton kernel that computes the real-to-complex FFT for a real input of length N,
    conceptually zero-padded to 2*N, and writes the first L_out = N//2 + 1 outputs.
    Each program computes one frequency index k in [0, L_out), and writes both real and imag parts.
    Normalization by 'scale' (1 / (2*seqlen)) is applied in-kernel.
    """
    # One program per output frequency index k
    pid = tl.program_id(axis=0)
    k = pid  # k in [0, L_out)

    # Accumulators for real and imag parts
    acc_real = 0.0
    acc_imag = 0.0

    # Zero-pad to 2*N conceptually: input length becomes 2*N, for n >= N, x[n] = 0
    TWO_N = 2 * N
    for n in range(0, TWO_N):
        # Mask to treat n >= N as zero (since we conceptually zero-pad)
        x_n = tl.load(x_ptr + n, mask=(n < N), other=0.0)
        angle = 2.0 * math.pi * k * n / TWO_N
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)
        acc_real += x_n * cos_term
        acc_imag += x_n * sin_term

    # Normalize by scale = 1 / (2*seqlen)
    y_real = acc_real * scale
    y_imag = acc_imag * scale

    # Store outputs
    tl.store(real_out_ptr + k, y_real)
    tl.store(imag_out_ptr + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect a single 3D tensor: (batch, channels, seqlen)
        if x.dim() != 3:
            raise ValueError(f"ModelNew expects a 3D tensor (batch, channels, seqlen). Got shape {tuple(x.shape)}")
        batch, channels, seqlen = x.shape

        # Cast to float32 (original code does this)
        x_f32 = x.to(torch.float32)

        # Effective input length N
        N = seqlen
        # Output length for rfft with n=2*N is L_out = (2*N)//2 + 1 = N + 1
        L_out = N // 2 + 1

        # Allocate outputs for real and imaginary parts: shape (batch, channels, L_out)
        device = x.device
        real_out = torch.empty((batch, channels, L_out), dtype=torch.float32, device=device)
        imag_out = torch.empty((batch, channels, L_out), dtype=torch.float32, device=device)

        # Scale factor for normalization
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernel: one program per k in [0, L_out)
        grid = (L_out,)
        _real_rfft_zero_pad_kernel[grid](
            x_f32.view(-1),  # flatten input to 1D
            real_out.view(-1),
            imag_out.view(-1),
            N, L_out, scale,
        )

        # Return real and imaginary parts; shape matches original (batch, channels, seqlen + 1)
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
