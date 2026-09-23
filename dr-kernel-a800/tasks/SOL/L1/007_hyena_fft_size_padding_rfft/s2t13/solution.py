import torch
import triton
import triton.language as tl

# Kernel A: zero-pad each (b, c) slice to length two_L = 2 * L into a flattened buffer.
# We allocate a buffer of length (B*C) * two_L and fill it with the first L elements from x and zeros for the next L.
@triton.jit
def pad_zero_kernel(x_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr, L: tl.constexpr):
    # One program per (b, c) slice
    bc = tl.program_id(0)  # 0 .. (B*C - 1)
    b = bc // C
    c = bc % C
    base_x = (b * C + c) * L
    base_out = bc * two_L

    # Copy first L elements
    for t in range(0, L):
        val = tl.load(x_ptr + base_x + t)
        tl.store(out_ptr + base_out + t, val)

    # Write zeros for the next L elements
    for t in range(0, L):
        tl.store(out_ptr + base_out + L + t, 0.0)

# Kernel B: compute real DFT for a specific (b, c, k). Assumes out_ptr contains the zero-padded vector of length 2*L for this (b, c).
# Output: stores real[k] and imag[k]. For real inputs, imag[k] should be zero, but we compute and store it.
@triton.jit
def real_dft_kernel(out_ptr, real_ptr, imag_ptr, k: tl.constexpr, L: tl.constexpr, two_L: tl.constexpr):
    acc_real = 0.0
    acc_imag = 0.0
    # Sum over all t in [0, 2*L)
    for t in range(0, two_L):
        x_t = tl.load(out_ptr + t)
        angle = 2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
        cos_t = tl.cos(angle)
        sin_t = tl.sin(angle)
        contrib_real = x_t * cos_t
        contrib_imag = -x_t * sin_t  # rfft contribution uses cos - i sin
        acc_real += contrib_real
        acc_imag += contrib_imag
    norm = 1.0 / float(two_L)
    tl.store(real_ptr + k, acc_real * norm)
    tl.store(imag_ptr + k, acc_imag * norm)

# Kernel C: zero-initialize the entire imaginary output buffer so that imag part is zeros for all k.
@triton.jit
def zero_imag_kernel(imag_ptr, size: tl.constexpr):
    # Single-program fill; caller can launch multiple programs for large sizes if needed.
    # Here we write zeros for the whole buffer using a simple loop.
    for i in range(0, size):
        tl.store(imag_ptr + i, 0.0)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of the original run function:
          - Input: x of shape (B, C, L), float32
          - Compute torch.fft.rfft(x, n=2*L) along last dim
          - Normalize by 2*L
          - Return real and imaginary parts, both float32, shape (B, C, L+1)
        """
        # Ensure input is float32 and contiguous
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        two_L = 2 * L
        out_len = L + 1

        # Allocate outputs: real and imaginary parts for all (b, c, k) flattened
        real_out = torch.empty(B * C * out_len, dtype=torch.float32, device=x.device)
        imag_out = torch.empty(B * C * out_len, dtype=torch.float32, device=x.device)

        # 1) Launch pad_zero_kernel to build padded inputs: shape (B*C, 2*L) flattened into a vector of length (B*C)*2*L
        M = B * C * two_L
        out_ptr = torch.empty(M, dtype=torch.float32, device=x.device)
        pad_zero_kernel[(B * C,)](x, out_ptr, B=B, C=C, L=L)

        # 2) Zero-initialize imaginary output buffer
        zero_imag_kernel[(1,)](imag_out, size=B * C * out_len)

        # 3) Launch real_dft_kernel for k in [0..L-1] to compute real parts; imag parts are already zeros.
        #    Grid has B*C*L programs: each program computes one k for one (b, c) pair.
        grid = (B * C * L,)
        real_dft_kernel[grid](out_ptr, real_out, imag_out, k=0, L=L, two_L=two_L)  # placeholder to initialize signature; we'll loop in Python

        # We need to run real_dft_kernel for each k. Triton launch with a Python loop:
        for bc in range(0, B * C):
            for k in range(0, L):
                real_dft_kernel[(1,)](out_ptr, real_out, imag_out, k=k, L=L, two_L=two_L)

        # Reshape outputs to (B, C, L+1)
        real_out = real_out.view(B, C, out_len)
        imag_out = imag_out.view(B, C, out_len)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
