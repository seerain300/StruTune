import torch
import triton
import triton.language as tl

# Triton kernel: for each (b, c), build zero-padded vector of length two_L = 2 * L
# Input: x_ptr pointing to x of shape (B, C, L), contiguous
# Output: out_pad_ptr of length (B*C) * two_L, where first L elements are x[b, c, :], next L zeros
@triton.jit
def pad_kernel(x_ptr, out_ptr, B: tl.int32, C: tl.int32, L: tl.int32, two_L: tl.int32):
    bc_id = tl.program_id(0)  # program id over B*C
    # Compute b, c from bc_id
    b = bc_id // C
    c = bc_id % C

    # Pointer to the start of x[b, c, :]
    x_base = x_ptr + (b * C + c) * L
    # Base output offset for this (b, c)
    out_base = bc_id * two_L

    # Copy x[b, c, :] into out_pad[0:L]
    offsets = tl.arange(0, L)
    vals = tl.load(x_base + offsets)  # masked by offsets < L; always true
    tl.store(out_ptr + out_base + offsets, vals)

    # Fill the remaining two_L - L positions with zeros
    pad_offsets = L + tl.arange(0, two_L - L)
    zeros = tl.zeros([two_L - L], dtype=tl.float32)
    tl.store(out_ptr + out_base + pad_offsets, zeros)

# Triton kernel: compute real DFT X[k] = sum_{t=0}^{two_L-1} x[t] * exp(-2*pi*i*k*t/two_L), k in [0..L]
# Input: out_pad_ptr of length M = (B*C) * two_L, contiguous
# Output: out_real_ptr of length M, will contain only real parts (imag part is zero)
@triton.jit
def real_dft_kernel(out_pad_ptr, out_real_ptr, out_imag_ptr, B: tl.int32, C: tl.int32, L: tl.int32, two_L: tl.int32):
    bc_id = tl.program_id(0)  # program id over B*C
    # Initialize accumulators for real and imaginary parts
    real_acc = tl.zeros((), dtype=tl.float32)
    imag_acc = tl.zeros((), dtype=tl.float32)

    # Precompute 2*pi / two_L as float
    two_pi_div_twoL = 2.0 * 3.141592653589793 / two_L

    # For each k in [0..L], accumulate sum over t in [0..2*L-1]
    # We will do this in Python for loop in Triton: Triton supports simple loops over compile-time constants.
    for k in range(0, L + 1):
        # Accumulate sum over t
        sum_real = tl.zeros((), dtype=tl.float32)
        sum_imag = tl.zeros((), dtype=tl.float32)
        for t in range(0, two_L):
            # Load x[t] from out_pad for this (b, c)
            # out_pad_ptr has layout: for each (b, c), a contiguous block of two_L elements
            # base offset for this (b, c) is bc_id * two_L
            x_val = tl.load(out_pad_ptr + (bc_id * two_L) + t)
            # Compute exp(-2*pi*i*k*t/two_L) = cos(...) - i*sin(...)
            angle = two_pi_div_twoL * k * t
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            # Multiply and accumulate
            sum_real += x_val * cos_term
            sum_imag -= x_val * sin_term  # i * sin is negative in (a + ib) accumulation
        # Store real part; imag part is zero for real inputs, but we can store zeros explicitly
        real_acc += sum_real
        imag_acc += sum_imag

    # Write results to output for this (b, c)
    out_base = bc_id * (L + 1)
    tl.store(out_real_ptr + out_base + 0, real_acc)
    # Imaginary part zeros
    tl.store(out_imag_ptr + out_base + 0, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 and contiguous
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        two_L = 2 * L
        M = B * C

        # Allocate padded buffer of length M * two_L
        out_pad = torch.empty(M * two_L, dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (b, c)
        pad_kernel[(B * C,)](x, out_pad, B, C, L, two_L, num_warps=4)

        # Allocate outputs: real and imag parts, shape (B, C, L+1) flattened
        out_real = torch.empty(M * (L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty(M * (L + 1), dtype=torch.float32, device=x.device)

        # Launch DFT kernel
        real_dft_kernel[(B * C,)](out_pad, out_real, out_imag, B, C, L, two_L, num_warps=4)

        # Reshape to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        # Normalize by 2*L for real outputs; imag remains zero for real inputs
        out_real = out_real / (2.0 * L)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
