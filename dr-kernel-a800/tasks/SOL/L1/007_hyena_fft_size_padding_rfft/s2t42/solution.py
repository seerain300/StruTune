import torch
import triton
import triton.language as tl

# Triton kernel: build zero-padded flattened input for all (b, c) slices.
# out_ptr: flattened output buffer of length M = (B*C) * (2*L)
# x_ptr: input x of shape (B, C, L), contiguous float32
@triton.jit
def pad_kernel(
    x_ptr,         # *float32
    out_ptr,       # *float32
    B, C, L, two_L,   # runtime integers
):
    pid = tl.program_id(0)  # one program per (b, c)
    b = pid // C
    c = pid % C

    # Base offset for this (b, c) slice in the flattened output
    base = (b * C + c) * two_L

    # Copy x[b, c, :] into out_ptr[base : base + L]
    for t in range(0, L):
        x_addr = (b * C + c) * L + t
        val = tl.load(x_ptr + x_addr)
        tl.store(out_ptr + base + t, val)

    # Write zeros for the next L positions
    for t in range(0, L):
        tl.store(out_ptr + base + L + t, 0.0)


# Triton kernel: compute DFT for each (b, c, k), k in [0..L], using the padded input.
# out_ptr: flattened padded input of length (B*C) * (2*L)
# out_real_ptr, out_imag_ptr: flattened output buffers of length (B*C) * (L+1)
@triton.jit
def dft_real_kernel(
    out_ptr,               # *float32
    out_real_ptr,          # *float32
    out_imag_ptr,          # *float32
    B, C, L, two_L,        # runtime integers
):
    # Grid dimension is set to B*C*(L+1); one program per (b, c, k)
    pid = tl.program_id(0)

    # Derive (b, c, k) from pid
    k = pid % (L + 1)
    tmp = pid // (L + 1)
    c = tmp % C
    b = tmp // C

    # Accumulate X[k] = sum_{t=0}^{2*L-1} x[t] * exp(-2πi k t / two_L)
    acc_real = 0.0
    acc_imag = 0.0

    for t in range(0, two_L):
        base = (b * C + c) * (2 * L)
        x_t = tl.load(out_ptr + base + t)
        angle = -2.0 * 3.141592653589793 * (k * t) / two_L
        coss = tl.cos(angle)
        sins = tl.sin(angle)
        acc_real += x_t * coss
        acc_imag += x_t * (-sins)

    # Normalize by two_L (original code divides by 2*L)
    norm = 1.0 / two_L
    acc_real = acc_real * norm
    acc_imag = acc_imag * norm

    # Store results to flattened output at index pid
    tl.store(out_real_ptr + pid, acc_real)
    tl.store(out_imag_ptr + pid, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure contiguous float32
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, C, L = x.shape
        two_L = 2 * L

        # Allocate flattened padded input buffer: (B*C) * (2*L)
        # No torch.zeros/cat; only torch.empty is allowed for allocation.
        out_pad = torch.empty((B * C * two_L,), device=x.device, dtype=torch.float32)

        # Launch pad_kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_kernel[grid_pad](
            x, out_pad, B, C, L, two_L,
            num_warps=1,
        )

        # Allocate 3D output buffers directly as (B, C, L+1); pass flattened pointers
        out_real_3d = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag_3d = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_real_flat = out_real_3d.view(-1)
        out_imag_flat = out_imag_3d.view(-1)

        # Launch dft_real_kernel: one program per (b, c, k), where k in [0..L]
        grid_dft = (B * C * (L + 1),)
        dft_real_kernel[grid_dft](
            out_pad, out_real_flat, out_imag_flat, B, C, L, two_L,
            num_warps=1,
        )

        # Return outputs directly; no torch.view needed beyond what Triton wrote into memory.
        # The 3D outputs are laid out contiguously, so returning them as is matches desired shape.
        return out_real_3d, out_imag_3d


def run(*args):
    return ModelNew()(*args)
