import torch
import triton
import triton.language as tl

@triton.jit
def real_dft_kernel(
    x_ptr,          # *float32, flattened input (B*C, 2*L) contiguous
    out_real_ptr,   # *float32, flattened output real (B*C, L+1)
    out_imag_ptr,   # *float32, flattened output imag (B*C, L+1), we fill zeros
    two_L: tl.constexpr,  # int, 2*L
    L_out: tl.constexpr,  # int, L+1
):
    # One program per (b*c) slice
    b_c = tl.program_id(0)  # index over B*C
    # Constants
    pi = 3.141592653589793
    inv_two_L = 1.0 / two_L

    # We will compute real part only; imaginary part for real inputs is zero
    # Loop over k in [0..L-1] -> output index is k+1 (since L_out = L + 1)
    # Note: Triton supports Python for-loops with constexpr limits.
    for k in range(L_out - 1):
        # Accumulate sum_{t=0}^{2*L-1} x[t] * cos(2*pi*k*t/(2*L))
        s = 0.0
        for t in range(two_L):
            x_t = tl.load(x_ptr + b_c * two_L + t)
            angle = 2.0 * pi * k * t * inv_two_L
            s += x_t * tl.cos(angle)
        # Normalize by 2*L to match original run: x_freq = x_freq / (2*L)
        s = s * inv_two_L  # this doubles the original division: s /= (2*L) -> s *= 1/(2*L)
        # Store real part at (b_c, k+1); imag part is zero
        tl.store(out_real_ptr + b_c * (L_out - 1) + k, s)
        tl.store(out_imag_ptr + b_c * (L_out - 1) + k, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor input: x of shape (B, C, L)
        x = args[0]
        # Ensure float32 contiguous
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        B, C, L = x.shape
        two_L = 2 * L
        L_out = L + 1

        # Build zero-padded input per (b, c) slice: [x, zeros(two_L - L)]
        # Shape: (B*C, 2*L), contiguous
        x_flat = x.view(B * C, L)
        pad = torch.zeros((B * C, two_L - L), dtype=torch.float32, device=x.device)
        x_padded = torch.cat([x_flat, pad], dim=1).contiguous()  # (B*C, 2*L)

        # Allocate outputs: (B*C, L+1)
        out_real = torch.empty((B * C, L_out), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, L_out), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b*c) slice
        grid = (B * C,)
        real_dft_kernel[grid](
            x_padded, out_real, out_imag,
            two_L=two_L, L_out=L_out
        )

        # Reshape to (B, C, L+1)
        out_real = out_real.view(B, C, L_out)
        out_imag = out_imag.view(B, C, L_out)

        # Normalize by 2*L to match original run: x_freq = x_freq / (2*L)
        # out_real already contains the sum; we applied inv_two_L in kernel.
        # out_imag is zeros.

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
