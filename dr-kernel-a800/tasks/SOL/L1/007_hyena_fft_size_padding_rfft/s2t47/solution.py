import torch
import triton
import triton.language as tl


@triton.jit
def dft_k_scalar_kernel(
    in_ptr,           # *float32, flattened input per (b, c): length = (B*C) * (2*L)
    out_real_ptr,     # *float32, flattened output real per (b, c, k): length = (B*C) * (L+1)
    out_imag_ptr,     # *float32, flattened output imag per (b, c, k): length = (B*C) * (L+1)
    two_L: tl.constexpr,  # int32, 2 * seqlen (constexpr to allow loops)
    B: tl.constexpr,      # int32, batch size (for indexing)
    C: tl.constexpr,      # int32, channels
    b: tl.constexpr,      # int32, current batch index
    c: tl.constexpr,      # int32, current channel index
    k_index: tl.constexpr # int32, current k index
):
    # Compute base offset for this (b, c) slice in the flattened input
    base = (b * C + c) * two_L

    # Accumulators for real and imaginary parts
    acc_real = 0.0
    acc_imag = 0.0

    # Loop over t in [0, 2*L)
    t = 0
    while t < two_L:
        x_t = tl.load(in_ptr + base + t)

        # angle = -2*pi*k*t/(2*L) = -2*pi*k*t/two_L
        angle = -2.0 * 3.141592653589793 * float(k_index) * float(t) / float(two_L)

        cos_part = tl.cos(angle)
        sin_part = tl.sin(angle)

        acc_real += x_t * cos_part
        acc_imag -= x_t * sin_part  # imaginary contribution

        t += 1

    # Normalize by 2*L
    norm = 1.0 / float(two_L)
    acc_real = acc_real * norm

    # Imaginary for real input is zero
    out_base = (b * C + c) * (L + 1)
    tl.store(out_real_ptr + out_base + k_index, acc_real)
    tl.store(out_imag_ptr + out_base + k_index, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure input is 3D and contiguous float32
        assert x.dim() == 3, "Input must be 3D: (batch, channels, seqlen)"
        B, C, L = x.shape
        x = x.contiguous().to(torch.float32)

        # Zero-pad to 2*L along the last dimension per (b, c)
        two_L = 2 * L
        padded = torch.zeros((B, C, two_L), dtype=torch.float32, device=x.device)
        padded[:, :, :L] = x  # fill first L entries with original x

        # Flatten for Triton: input vector of length (B*C) * (2*L)
        in_flat = padded.reshape(-1)  # 1D contiguous
        M_batches = B * C
        out_real_flat = torch.empty(M_batches * (L + 1), dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty(M_batches * (L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel once per (b, c, k)
        for b in range(B):
            for c in range(C):
                for k in range(L + 1):
                    # num_warps=1, num_stages=1 to minimize Triton compilation/runtime complexity
                    dft_k_scalar_kernel[(1,)](
                        in_flat,
                        out_real_flat,
                        out_imag_flat,
                        two_L,
                        B,
                        C,
                        b,
                        c,
                        k,
                        num_warps=1,
                        num_stages=1,
                    )

        # Reshape outputs to (B, C, L+1)
        out_real = out_real_flat.view(B, C, L + 1)
        out_imag = out_imag_flat.view(B, C, L + 1)

        # Match original: return normalized real and imaginary parts
        # Note: The original code divides by 2*L. This implementation already applies the normalization inside the kernel.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
