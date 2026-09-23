import torch
import triton
import triton.language as tl


# Triton kernel: build zero-padded flattened input for each (b, c) slice.
# We flatten all (B*C) slices into a single vector of length total_elems = (B*C) * (2*L).
# Each program handles one (b, c) slice and writes its L + zeros into out_ptr at offset start.
@triton.jit
def pad_flat_kernel(x_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr, L: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1)
    # total_elems = (B*C) * (2*L)
    # start index in out_ptr for this (b, c) slice
    start = (b * C + c) * (2 * L)
    # Copy x[b, c, :] of length L into out_ptr[start : start + L]
    # Note: Triton requires pointer arithmetic to be simple and known at compile time per program.
    for t in range(0, L):
        val = tl.load(x_ptr + b * C * L + c * L + t)
        tl.store(out_ptr + start + t, val)
    # Write zeros for the remaining L positions
    for t in range(0, L):
        tl.store(out_ptr + start + L + t, 0.0)


# Triton kernel: compute real DFT for a single (b, c) slice and a single k.
# in_ptr is a flattened vector of length two_L for (b, c). out_real_ptr/out_imag_ptr are flattened of length L+1.
@triton.jit
def real_dft_scalar_kernel(in_ptr, out_real_ptr, out_imag_ptr,
                           B: tl.constexpr, C: tl.constexpr,
                           L: tl.constexpr, two_L: tl.constexpr,
                           k: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1)
    base_out = b * C * (L + 1) + c * (L + 1)

    acc_real = 0.0
    acc_imag = 0.0

    for t in range(0, two_L):
        val = tl.load(in_ptr + t)
        angle = (2.0 * 3.141592653589793 * k * t) / two_L
        acc_real += val * tl.cos(angle)
        acc_imag += -val * tl.sin(angle)

    norm = 1.0 / two_L
    acc_real = acc_real * norm
    acc_imag = acc_imag * norm

    tl.store(out_real_ptr + base_out + k, acc_real)
    tl.store(out_imag_ptr + base_out + k, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x is (B, C, L), we assume float32 and contiguous. No torch ops allowed in forward.
        B, C, L = x.shape
        two_L = 2 * L

        # We must build the padded flattened input vector without torch ops.
        # Allocate out buffer of length (B*C) * (2*L), zeros. Triton will write the first L values per (b,c) from x.
        total_elems = (B * C) * (2 * L)
        out_flat = torch.empty(total_elems, dtype=torch.float32, device=x.device)

        # Launch pad_flat_kernel: one program per (b, c)
        grid = (B, C)
        pad_flat_kernel[grid](x, out_flat, B, C, L)

        # Allocate outputs: real and imag, flattened as (B*C, L+1)
        out_real = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)

        # Launch real DFT kernel: grid (B, C), loop over k on host
        for k in range(0, L + 1):
            real_dft_scalar_kernel[grid](out_flat, out_real, out_imag, B, C, L, two_L, k)

        # Reshape outputs to (B, C, L+1). Note: torch.view is allowed here since it's host-side reshaping metadata.
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
