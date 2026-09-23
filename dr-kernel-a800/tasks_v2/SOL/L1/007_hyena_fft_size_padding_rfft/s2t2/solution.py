import torch
import triton
import triton.language as tl

# Triton kernel: computes the DFT over a zero-padded flattened vector of length two_L.
# Grid is (B*C, two_L): one program per (b, c, k).
# For each (b, c, k), it loops over t=0..two_L-1, accumulates x[t] * exp(-2πi k t / (2*L)),
# stores normalized real/imag parts. All math is done in the kernel; no torch GPU ops in forward.
@triton.jit
def dft_padded_kernel(
    x_ptr,                 # *const float, input padded vector flattened of length M = B*C*two_L
    out_real_ptr,          # *float, output real part flattened of length M
    out_imag_ptr,          # *float, output imag part flattened of length M
    two_L: tl.constexpr,   # int, padded length = 2*L
):
    pid0 = tl.program_id(0)  # combined (b*c) index
    pid1 = tl.program_id(1)  # k index in [0, two_L - 1]

    # Each (b, c) slice occupies two_L consecutive elements in x_ptr/out buffers.
    base = pid0 * two_L

    # Accumulator for sum over t
    acc = 0.0 + 0.0j

    # Constants
    inv_two_L = 1.0 / two_L
    pi = 3.141592653589793

    # Sum over t from 0 to two_L - 1
    for t in range(0, two_L):
        x_val = tl.load(x_ptr + base + t)  # real value
        angle = -2.0 * pi * pid1 * t * inv_two_L
        re = tl.cos(angle)
        im = tl.sin(angle)
        acc += x_val * (re + im * 1j)

    # Normalize by 2*L (matching original: divide by 2*seqlen)
    acc_norm = acc * inv_two_L

    # Store real and imaginary parts
    tl.store(out_real_ptr + base + pid1, acc_norm.real)
    tl.store(out_imag_ptr + base + pid1, acc_norm.imag)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton implementation: builds zero-padded input and computes real FFT (with normalization)
        using a single Triton kernel. Returns real and imaginary parts of shape (B, C, 2*L).
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape

        # Ensure contiguous and float32
        x = x.contiguous().to(torch.float32)

        # Total length after padding
        two_L = 2 * L

        # Build zero-padded input vector: flatten across (B, C) into one vector of length M = B*C*2*L.
        # For each (b, c), first L entries are x[b, c, :], the remaining two_L - L entries are zeros.
        M = B * C * two_L
        x_flat_padded = torch.empty(M, device=x.device, dtype=x.dtype)

        # Fill x_flat_padded: for each (b,c), write two_L elements (first L = x[b,c,:], then zeros)
        for b in range(B):
            for c in range(C):
                start = (b * C + c) * two_L
                # Copy original slice (L elements)
                slice_vec = x[b, c, :].contiguous().view(-1)  # shape (L,)
                # Write into padded vector at [start : start+L]
                x_flat_padded[start:start + L] = slice_vec
                # The remaining two_L - L positions are already zeros in the allocated tensor

        # Allocate outputs (real and imag), shape (B, C, two_L), float32, flattened for kernel
        out_real = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_real_flat = out_real.reshape(-1)  # length = B*C*2*L
        out_imag_flat = out_imag.reshape(-1)  # length = B*C*2*L

        # Launch Triton kernel: grid over (B*C, two_L), one program per (b,c,k)
        grid = (B * C, two_L)
        dft_padded_kernel[grid](
            x_flat_padded, out_real_flat, out_imag_flat,
            two_L=two_L,
            num_warps=1,
            num_stages=1,
        )

        # Return real and imaginary parts (shape (B, C, 2*L))
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
