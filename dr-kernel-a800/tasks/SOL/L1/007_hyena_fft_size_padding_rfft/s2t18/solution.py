import torch
import triton
import triton.language as tl

# Triton kernel: For each (b, c), compute real DFT across k in [0..L] of the padded vector (length = 2*L).
# Uses scalar loops to avoid Triton address arithmetic issues.
@triton.jit
def real_dft_scalar_kernel(
    pad_ptr,           # *float32, padded input vector for all (b,c) flattened
    out_real_ptr,      # *float32, output real part flattened
    out_imag_ptr,      # *float32, output imaginary part flattened (zeros)
    L,                 # int32, original sequence length
    two_L,             # int32, padded length = 2*L
    BC,                # int32, total number of (b,c) slices = B*C
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    idx = b * C + c

    # Base offset for this (b, c) slice in the output
    base_out = idx * (L + 1)

    # Loop over k = 0..L-1
    for k in range(0, L):
        sum_real = 0.0
        sum_imag = 0.0

        # Compute DFT for this k over the 2*L padded vector
        for t in range(0, two_L):
            # Load padded[idx*two_L + t]
            val = tl.load(pad_ptr + idx * two_L + t)
            # Compute angle = -2π * k * t / (2*L)
            angle = -2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
            ct = tl.cos(angle)
            st = tl.sin(angle)
            # For real inputs, final imaginary part should be zero; we still accumulate imag for correctness.
            sum_real += val * ct
            sum_imag += -val * st  # negative sign because of -i in DFT definition

        # Normalize by 2*L (matches original normalization)
        sum_real = sum_real / float(two_L)
        sum_imag = sum_imag / float(two_L)  # zero for real inputs

        # Store real and imag parts to output
        tl.store(out_real_ptr + base_out + k, sum_real)
        tl.store(out_imag_ptr + base_out + k, sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute outputs equivalent to:
        - x_f32 = x.to(float32)
        - x_freq = torch.fft.rfft(x_f32, n=2*L)
        - x_freq = x_freq / (2*L)
        - return x_freq.real, x_freq.imag  # both shape (B, C, L+1), float32
        Using Triton for the heavy computation.
        """
        # Ensure float32 and contiguous
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        B, C, L = x.shape
        two_L = 2 * L

        # Prepare padded input for each (b, c): [x, zeros(two_L - L)]
        # Flatten all (b,c) slices into a single vector of length BC * two_L
        BC = B * C
        pad_vec = torch.empty((BC, two_L), dtype=torch.float32, device=x.device)
        idx = 0
        for b in range(B):
            for c in range(C):
                # Place x[b, c, :] at the beginning, zeros for the rest
                pad_vec[idx, :L] = x[b, c, :]
                pad_vec[idx, L:] = 0.0
                idx += 1

        # Allocate outputs (flattened per (b, c) slice)
        out_real_flat = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: grid over (B, C)
        real_dft_scalar_kernel[(B, C)](
            pad_vec, out_real_flat, out_imag_flat,
            L, two_L, BC,
        )

        # Reshape back to (B, C, L+1)
        out_real = out_real_flat.view(B, C, L + 1)
        out_imag = out_imag_flat.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
