import torch
import triton
import triton.language as tl


# Triton kernel: build zero-padded input vector for all (b, c) slices flattened.
# Input x: shape (B, C, L), contiguous float32 on GPU.
# Output padded: flattened vector of length M = (B*C)*(2*L), where padded[base + t] = x[b,c,t] for t in [0, L),
# and padded[base + t + L] = 0 for t in [0, L-1]. base = (b*C + c)*2*L.
@triton.jit
def pad_kernel(
    x_ptr,          # *const float, input tensor pointer
    padded_ptr,     # *float, output flattened padded vector pointer
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    two_L: tl.constexpr,
    # grid: (B*C,)
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    base = (b * C + c) * two_L  # starting offset for this (b, c) slice in the flattened padded vector

    # Copy x[b, c, :] into the first L positions
    t_offsets = tl.arange(0, two_L)                 # vector [0, 1, ..., 2*L-1]
    valid_t = t_offsets < L                         # mask for valid x positions
    x_offsets = b * C * L + c * L + t_offsets      # input offsets within x
    x_vals = tl.load(x_ptr + x_offsets, mask=valid_t, other=0.0)
    tl.store(padded_ptr + base + t_offsets, x_vals, mask=valid_t)

    # Zero the padded part [L, 2*L)
    padded_offsets = t_offsets + L
    tl.store(padded_ptr + base + padded_offsets, 0.0, mask=valid_t)


# Triton kernel: compute real DFT over zero-padded input vector, normalize by 2*L,
# and write real and imaginary parts. Grid over (B, C, 2*L).
@triton.jit
def real_dft_zero_padded_kernel(
    padded_ptr,     # *const float, flattened zero-padded input vector
    out_real_ptr,   # *float, output real part (flattened length M)
    out_imag_ptr,   # *float, output imag part (flattened length M)
    B: tl.constexpr,
    C: tl.constexpr,
    two_L: tl.constexpr,
    inv_two_L,      # float32, 1.0 / (2*L)
    # grid: (B, C, two_L)
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.program_id(2)

    # Compute base offset in flattened padded vector for this (b, c)
    base = (b * C + c) * two_L

    # Accumulators for real and imaginary parts
    accum_real = tl.zeros((), dtype=tl.float32)
    accum_imag = tl.zeros((), dtype=tl.float32)

    # Loop over t = 0..2*L-1: load x_t from padded vector and accumulate
    for t in range(0, two_L):
        x_t = tl.load(padded_ptr + base + t)  # scalar load
        angle = -2.0 * 3.141592653589793 * k * t * inv_two_L  # -2*pi*k*t / (2*L)
        real_part = tl.cos(angle)
        imag_part = tl.sin(angle)
        # Accumulate: x_t * (cos(angle) - i*sin(angle))
        accum_real = accum_real + x_t * real_part
        accum_imag = accum_imag - x_t * imag_part

    # Normalize by 2*L
    accum_real = accum_real * inv_two_L
    accum_imag = accum_imag * inv_two_L

    # Store results into flattened output at positions corresponding to (b, c, k)
    out_base = (b * C + c) * two_L + k  # linear index where real and imag are stored
    tl.store(out_real_ptr + out_base, accum_real)
    tl.store(out_imag_ptr + out_base, accum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L), float32, on GPU (Triton requires CUDA tensors)
        assert x.is_cuda, "Input must be a CUDA tensor."
        B, C, L = x.shape
        two_L = 2 * L

        # Ensure x is contiguous float32
        x = x.contiguous().to(torch.float32)

        # Allocate flattened padded input vector: length M = (B*C)*(2*L)
        M = (B * C) * two_L
        padded = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_kernel[grid_pad](
            x, padded,
            B, C, L, two_L,
            num_warps=1,
        )

        # Allocate outputs (flattened)
        out_real = torch.empty(M, dtype=torch.float32, device=x.device)
        out_imag = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch real DFT kernel: grid over (B, C, 2*L)
        grid_dft = (B, C, two_L)
        inv_two_L = 1.0 / float(two_L)
        real_dft_zero_padded_kernel[grid_dft](
            padded, out_real, out_imag,
            B, C, two_L, inv_two_L,
            num_warps=4,
        )

        # Reshape back to (B, C, 2*L)
        x_freq_real = out_real.view(B, C, two_L)
        x_freq_imag = out_imag.view(B, C, two_L)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
