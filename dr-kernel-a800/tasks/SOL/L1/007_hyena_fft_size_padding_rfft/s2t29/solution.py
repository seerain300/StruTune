import torch
import triton
import triton.language as tl

# Triton kernel: zero-pad each (b, c) slice into a vector of length two_L = 2 * L.
# Each program handles one (b, c) slice.
@triton.jit
def pad_kernel(
    x_flat_ptr,         # *f32, pointer to flattened input x, shape (BC, L)
    out_ptr,            # *f32, pointer to output padded vector, shape (BC, two_L)
    L,                  # int, seqlen
    two_L,              # int, 2 * seqlen
    BC,                 # int, total number of (b,c) slices = B*C
    b_idx,              # int, current b index for this program
    c_idx,              # int, current c index for this program
):
    # Compute linear index for the (b, c) slice in x_flat
    bc_idx = b_idx * C + c_idx
    # Base pointer for this slice
    base = bc_idx * L

    # Copy first L elements from x_flat into out
    for t in range(0, L):
        val = tl.load(x_flat_ptr + base + t)
        tl.store(out_ptr + bc_idx * two_L + t, val)

    # Fill remaining L elements with zeros
    for t in range(L, two_L):
        tl.store(out_ptr + bc_idx * two_L + t, 0.0)


# Triton kernel: compute real DFT over the padded vector and write normalized real part.
# One program per (bc, k) where bc in [0..BC-1], k in [0..L].
@triton.jit
def dft_real_kernel_padded(
    x_padded_ptr,       # *f32, pointer to padded vector, shape (BC, two_L)
    out_real_ptr,       # *f32, pointer to output real part, shape (BC, L+1)
    out_imag_ptr,       # *f32, pointer to output imag part, shape (BC, L+1)
    L,                  # int, seqlen
    two_L,              # int, 2 * seqlen
    BC,                 # int, total number of (b,c) slices
    bc_idx,             # int, current (b,c) slice index
    k,                  # int, current frequency index
):
    # Accumulator for real part
    acc = 0.0

    # Iterate over all t in [0, 2*L-1]
    for t in range(0, two_L):
        x_t = tl.load(x_padded_ptr + bc_idx * two_L + t)
        # angle = -2*pi*k*t / (2*L) = -pi*k*t / L
        angle = -3.141592653589793 * k * t / L
        e = tl.exp(1j * angle)
        # Since x_t is real, x_t * e is complex. We only need the real part contribution:
        # real(x_t * e) = x_t * cos(angle) ; imag part is irrelevant for real rfft output.
        # However, to keep computation fully in Triton, we compute the full complex product and take real.
        # Note: Triton may not support direct complex math uniformly; we compute cos(angle) and multiply.
        cos_angle = tl.cos(angle)
        acc += x_t * cos_angle

    # Normalize by 2*L
    acc = acc / two_L

    # Store real and imag parts (imag is zero for real inputs)
    tl.store(out_real_ptr + bc_idx * (L + 1) + k, acc)
    tl.store(out_imag_ptr + bc_idx * (L + 1) + k, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure dtype and contiguity
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, C, L = x.shape
        two_L = 2 * L
        BC = B * C

        # Flatten x to (BC, L) for simpler Triton indexing
        x_flat = x.view(BC, L)

        # Allocate padded vector: zeros
        out_padded = torch.zeros((BC, two_L), device=x.device, dtype=torch.float32)

        # Launch pad_kernel for each (b, c)
        grid_pad = (BC,)
        pad_kernel[grid_pad](x_flat_ptr=x_flat, out_ptr=out_padded, L=L, two_L=two_L, BC=BC, b_idx=0, c_idx=0, num_warps=1)

        # Allocate outputs (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Launch dft_real_kernel_padded: one program per (bc, k) where k in [0..L]
        grid_dft = (BC, L + 1)
        dft_real_kernel_padded[grid_dft](x_padded_ptr=out_padded, out_real_ptr=out_real, out_imag_ptr=out_imag, L=L, two_L=two_L, BC=BC, bc_idx=0, k=0, num_warps=1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
