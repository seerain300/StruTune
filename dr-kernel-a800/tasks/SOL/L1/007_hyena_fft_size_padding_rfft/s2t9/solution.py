import math
import torch
import triton
import triton.language as tl

# Triton kernel: compute real DFT over zero-padded vector of length two_L.
# Grid: (B, C, L+1). Each program handles one (b, c, k).
@triton.jit
def dft_real_padded_kernel(
    x_ptr,             # *float32, flattened zero-padded input vector for (B*C) slices
    out_real_ptr,      # *float32, output real part flattened
    out_imag_ptr,      # *float32, output imag part flattened (should be zeros for real input)
    B: tl.constexpr,   # batch size
    C: tl.constexpr,   # channels
    L: tl.constexpr,   # original seqlen
    two_L: tl.constexpr,  # padded length = 2 * L
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)  # k index in [0, L]

    # Base offset for this (b, c) slice in the flattened input/output buffers.
    bc = pid_b * C + pid_c
    base = bc * two_L

    # Accumulator for this k
    accum = 0.0  # Triton will treat this as float32

    # Loop over all t in [0, 2*L-1]
    # x_t = x_ptr[base + t] for t < L, else 0. We use mask to avoid loading non-existent t (though pointer arithmetic is fine).
    for t in range(0, two_L):
        x_t = tl.load(x_ptr + base + t)
        angle = -2.0 * math.pi * float(pid_k) * float(t) / float(two_L)
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)
        # exp(-i*angle) = cos(angle) - i*sin(angle). For real inputs, imag contribution should be zero anyway.
        # Compute contribution for real part: x_t * cos(angle), imaginary part: -x_t * sin(angle)
        # But since input is real, rfft imaginary part is zero; we set imag to zero.
        accum += x_t * cos_term

    # Normalize by 2*L
    inv_two_L = 1.0 / float(two_L)
    real_val = accum * inv_two_L

    # Write outputs
    out_real_offset = bc * (L + 1) + pid_k
    out_imag_offset = bc * (L + 1) + pid_k
    tl.store(out_real_ptr + out_real_offset, real_val)
    tl.store(out_imag_ptr + out_imag_offset, 0.0)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L), float32
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        assert x.dtype == torch.float32, "Input must be float32"
        assert x.ndim == 3, "Input must be 3D (B, C, L)"
        B, C, L = x.shape
        two_L = 2 * L

        # Ensure contiguous
        x = x.contiguous()

        # Build zero-padded input per (b, c) slice: [x, zeros]
        # We flatten (B*C) slices into one vector of length M = (B*C) * two_L
        x_2d = x.view(B, C, L)            # for view, already contiguous
        # Create zeros tensor of shape (B, C, two_L - L)
        zeros_bc = torch.zeros((B, C, two_L - L), dtype=torch.float32, device=x.device)
        # Concatenate along last dim to form padded vectors
        padded_bc = torch.cat([x_2d, zeros_bc], dim=2)  # shape (B, C, two_L)
        # Flatten to 1D for Triton
        padded_flat = padded_bc.reshape(-1)  # length M = (B*C) * two_L

        # Allocate outputs (flattened)
        out_real = torch.empty((B * C) * (L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C) * (L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c, k) with k in [0..L]
        grid = (B, C, L + 1)
        dft_real_padded_kernel[grid](
            padded_flat, out_real, out_imag,
            B, C, L, two_L,
            num_warps=4,
        )

        # Reshape to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
