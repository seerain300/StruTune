import torch
import triton
import triton.language as tl

# Triton kernel: for each (b, c, k), compute the DFT over a zero-padded input vector of length two_L.
# Grid is (B, C, two_L): one program per (b, c, k). It loops over t in [0, two_L-1], loads x[t]
# (from the flattened padded vector), computes exp(-2πi k t / (2*L)), accumulates, normalizes by 2*L,
# and stores real/imag parts into flattened outputs.
@triton.jit
def real_dft_zero_padded_kernel(
    x_ptr,                 # *const float, input padded vector flattened of length two_L
    out_real_ptr,          # *float, output real part flattened of length two_L
    out_imag_ptr,          # *float, output imag part flattened of length two_L
    two_L: tl.constexpr,   # int, padded length = 2 * L
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)  # k index in [0, two_L - 1]

    # Flatten (B, C) into a single dimension for pointer indexing
    bc = pid_b * C + pid_c
    base = bc * two_L  # each (b, c) slice occupies two_L consecutive elements in the flattened vector

    # Accumulator for the sum over t
    acc = 0.0 + 0.0j

    # Constants
    inv_two_L = 1.0 / two_L
    pi = 3.141592653589793

    # Sum over t from 0 to two_L - 1
    for t in range(0, two_L):
        x_val = tl.load(x_ptr + base + t)  # x_val is real
        angle = -2.0 * pi * pid_k * t * inv_two_L
        re = tl.cos(angle)
        im = tl.sin(angle)
        acc += x_val * (re + im * 1j)

    # Normalize by 2*L (matching original: divide by 2*seqlen)
    acc_norm = acc * inv_two_L

    # Store real and imaginary parts
    tl.store(out_real_ptr + base + pid_k, acc_norm.real)
    tl.store(out_imag_ptr + base + pid_k, acc_norm.imag)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton implementation of the original run function.
        Computes real FFT with zero-padding to 2*seqlen using a Triton kernel,
        and returns real and imaginary parts normalized by 2*L.
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape

        # Ensure contiguous and float32
        x = x.contiguous().to(torch.float32)

        # Total length after padding
        two_L = 2 * L

        # Build zero-padded input vector: flatten all (b, c) slices into one vector of length M = B*C*two_L.
        # For each (b, c), first L entries are x[b, c, :], the remaining two_L - L entries are zeros.
        M = B * C * two_L
        x_flat_padded = torch.empty(M, device=x.device, dtype=x.dtype)

        # Fill x_flat_padded: for each (b,c), write two_L elements (first L = x[b,c,:], then zeros)
        for b in range(B):
            for c in range(C):
                start = (b * C + c) * two_L
                slice_vec = x[b, c, :].contiguous()  # shape (L,)
                zeros_vec = torch.zeros(two_L - L, device=x.device, dtype=x.dtype)
                x_flat_padded[start:start + two_L] = torch.cat([slice_vec, zeros_vec])

        # Allocate outputs (real and imag), shape (B, C, two_L), float32, flattened
        out_real = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_real_flat = out_real.reshape(-1)  # length = B*C*2*L
        out_imag_flat = out_imag.reshape(-1)  # length = B*C*2*L

        # Launch Triton kernel: grid over (B, C, two_L), one program per (b,c,k)
        grid = (B, C, two_L)

        real_dft_zero_padded_kernel[grid](
            x_flat_padded, out_real_flat, out_imag_flat,
            two_L=two_L,
            num_warps=1,
            num_stages=1,
        )

        # Return real and imaginary parts as requested (shape (B, C, 2*L))
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
