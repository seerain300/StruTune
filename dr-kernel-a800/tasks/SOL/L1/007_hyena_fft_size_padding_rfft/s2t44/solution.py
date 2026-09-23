import torch
import triton
import triton.language as tl

# Triton kernel: zero-pad each (b, c) slice into a flattened buffer of length two_L = 2*L
# x_ptr: *float32, input x with shape (B, C, L), contiguous
# out_ptr: *float32, output flattened padded vector with length M = (B*C) * two_L, contiguous
# B, C: runtime ints (not constexpr), L is passed as L
@triton.jit
def pad_kernel(x_ptr, out_ptr, B, C, L, two_L):
    # program id over flattened (b, c)
    pid = tl.program_id(0)
    # compute (b, c) from pid
    b = pid // C
    c = pid % C
    # base offset in the flattened out buffer for this (b, c)
    base = pid * two_L
    # copy x[b, c, :] into the first L positions
    # Triton allows simple scalar loops for robustness across versions
    for t in range(0, L):
        val = tl.load(x_ptr + b * C * L + c * L + t)  # address: ((b*C + c)*L + t)
        tl.store(out_ptr + base + t, val)
    # fill the next L positions with zeros
    for t in range(L, two_L):
        tl.store(out_ptr + base + t, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure input is float32 and contiguous
        assert x.dim() == 3, "Input must be (B, C, L)"
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        two_L = 2 * L

        # Allocate flattened padded buffer: (B*C) * (2*L)
        M = B * C * two_L
        padded = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch Triton padding kernel: one program per (b, c)
        grid = (B * C,)
        # Note: passing B, C, L, two_L as runtime ints
        pad_kernel[grid](x, padded, B, C, L, two_L)

        # Reshape padded buffer to (B, C, 2*L) for torch.rfft
        # The order is such that each (b, c) slice is the first L elements plus L zeros.
        x_padded = padded.view(B, C, two_L)

        # Compute real FFT along the last dimension, length n=two_L
        # Output is complex, length (two_L//2 + 1) = (2*L)//2 + 1 = L+1
        # For real inputs, torch.fft.rfft returns exact complex output.
        x_freq = torch.fft.rfft(x_padded, n=two_L)

        # Normalize by 2*L
        x_freq = x_freq / (2 * L)

        # Extract real and imaginary parts, reshape to (B, C, L+1)
        x_freq_real = x_freq.real.contiguous().view(B, C, L + 1)
        x_freq_imag = x_freq.imag.contiguous().view(B, C, L + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
