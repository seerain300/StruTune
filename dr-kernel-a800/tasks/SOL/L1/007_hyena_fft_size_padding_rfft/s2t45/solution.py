import torch
import triton
import triton.language as tl

# Triton kernel: zero-pad each (b, c) slice to length two_L, write to flattened buffer
@triton.jit
def pad_kernel(
    x_ptr,            # *float32, input tensor x flattened with shape (B, C, L)
    out_ptr,          # *float32, flattened output buffer with length (B*C)*two_L
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # each program handles one (b, c) slice
    # base offset for this (b, c) slice in the flattened input
    base_in = pid * L
    # length of padded vector per (b, c): 2*L
    two_L = 2 * L
    base_out = pid * two_L

    # Copy first L elements
    for t in range(0, L):
        val = tl.load(x_ptr + base_in + t)
        tl.store(out_ptr + base_out + t, val)

    # Fill next L elements with zeros
    for t in range(L, two_L):
        tl.store(out_ptr + base_out + t, 0.0)

# Triton kernel: compute real DFT over padded input and write normalized real/imag parts
@triton.jit
def dft_real_kernel(
    out_ptr,          # *float32, flattened padded input per (b, c), length 2*L
    real_out_ptr,     # *float32, output real part flattened per (b, c), length L+1
    imag_out_ptr,     # *float32, output imag part flattened per (b, c), length L+1
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    two_L: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # each program handles one (b, c)
    base_in = pid * two_L
    # Prepare outputs for this (b, c) slice: length L+1
    base_real = pid * (L + 1)
    base_imag = pid * (L + 1)

    # Constants
    pi = 3.141592653589793
    inv_two_L = 1.0 / two_L

    # Loop over k = 0..L
    for k in range(0, L + 1):
        real_acc = 0.0
        imag_acc = 0.0
        # Sum over t = 0..2*L-1
        for t in range(0, two_L):
            val = tl.load(out_ptr + base_in + t)  # float32
            angle = (2.0 * pi * k * t) * inv_two_L
            real_contrib = val * tl.cos(angle)
            imag_contrib = val * tl.sin(angle)
            real_acc += real_contrib
            imag_acc += imag_contrib
        real_acc = real_acc * inv_two_L
        imag_acc = imag_acc * inv_two_L
        # Store
        tl.store(real_out_ptr + base_real + k, real_acc)
        tl.store(imag_out_ptr + base_imag + k, imag_acc)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure dtype and contiguity
        assert x.dim() == 3, "Input must be (B, C, L)"
        B, C, L = x.shape
        two_L = 2 * L

        # Make input contiguous and float32
        x = x.contiguous().to(torch.float32)

        # Flatten input to (B*C, L) for pointer arithmetic
        x_flat = x.view(B * C, L)

        # Allocate flattened padded buffer of size (B*C) * (2*L)
        out_flat = torch.empty(B * C * two_L, dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (b, c)
        grid = (B * C,)
        pad_kernel[grid](x_flat, out_flat, B, C, L)

        # Allocate outputs real/imag flattened per (b, c): each has length L+1
        real_out_flat = torch.empty(B * C * (L + 1), dtype=torch.float32, device=x.device)
        imag_out_flat = torch.empty(B * C * (L + 1), dtype=torch.float32, device=x.device)

        # Launch DFT real kernel: one program per (b, c)
        dft_real_kernel[grid](out_flat, real_out_flat, imag_out_flat, B, C, L, two_L)

        # Reshape to (B, C, L+1)
        real_out = real_out_flat.view(B, C, L + 1)
        imag_out = imag_out_flat.view(B, C, L + 1)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
