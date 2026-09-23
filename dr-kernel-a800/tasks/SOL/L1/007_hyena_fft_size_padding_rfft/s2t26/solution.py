import math
import torch
import triton
import triton.language as tl


@triton.jit
def real_dft_scalar_kernel(
    x_ptr,           # *f32, flattened padded input buffer (length = two_L), contiguous
    out_ptr,         # *f32, flattened output buffer (length = B*C*(L+1)), contiguous
    B: tl.constexpr, # int
    C: tl.constexpr, # int
    L: tl.constexpr, # int
    two_L: tl.constexpr, # int = 2*L
    # grid = (B*C, L+1), one program per (bc, k)
):
    bc = tl.program_id(0)   # 0..B*C-1
    k = tl.program_id(1)    # 0..L

    # base offset in flattened outputs: out layout is (bc, 0..L)
    base = bc * (L + 1) + k

    # Accumulator for X[k] as float32
    acc = 0.0

    # Sum over t from 0 to 2*L - 1: for padded input, x[t] = 0 for t >= L
    for t in range(0, two_L):
        x_t = tl.load(x_ptr + t)  # x_t is real float32
        # Angle in radians: 2*pi*k*t / (2*L)
        angle = (2.0 * math.pi * k * t) / two_L
        cos_angle = tl.cos(angle)
        acc += x_t * cos_angle

    # Normalize by 2*L
    acc = acc / two_L

    # Store real part; imaginary part is zero (for real input)
    tl.store(out_ptr + base, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Pads input per (b, c) to length 2*L with zeros.
        - Computes real DFT per (b, c, k) for k in [0..L], normalizes by 2*L.
        - Returns real and imaginary parts of shape (B, C, L+1) as float32 tensors.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton."
        assert x.dtype == torch.float32, "Input must be float32."
        x = x.contiguous()
        B, C, L = x.shape
        two_L = 2 * L

        # Build zero-padded input for each (b, c): concatenate x[b, c, :] with zeros of length L
        # Flatten across B and C for simple pointer arithmetic
        x_flat = x.view(B * C, L).contiguous()
        zeros = torch.zeros((B * C, L), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x_flat, zeros], dim=1).contiguous()  # shape (B*C, 2*L)

        # Prepare outputs: we only compute the real part; imaginary is zero.
        out = torch.empty((B * C, L + 1), device=x.device, dtype=torch.float32)

        # Launch Triton kernel: one program per (bc, k)
        grid = (B * C, L + 1)
        # num_warps can be tuned; use 1 for scalar work
        real_dft_scalar_kernel[grid](
            x_padded, out,
            B, C, L, two_L,
            num_warps=1,
        )

        # Reshape to (B, C, L+1)
        x_freq_real = out.view(B, C, L + 1)
        # Imaginary part is zero for real input; match original: create zeros tensor
        x_freq_imag = torch.zeros((B, C, L + 1), device=x.device, dtype=torch.float32)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
