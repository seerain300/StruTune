import torch
import triton
import triton.language as tl


@triton.jit
def pad_input_kernel(x_ptr, in_ptr,
                      B: tl.constexpr, C: tl.constexpr, L: tl.constexpr):
    # One Triton program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)
    # Source base offset for (b, c): x has shape (B, C, L) -> flatten as (B*C, L)
    base = b * C * L + c * L
    # Destination base offset for flattened (B*C, 2*L)
    dest_bc = b * C * (2 * L) + c * (2 * L)
    # Copy x[b, c, :] into in_ptr at positions [0..L-1]
    for t in range(0, L):
        val = tl.load(x_ptr + base + t)
        tl.store(in_ptr + dest_bc + t, val)
    # Fill the remaining two_L - L positions with zeros
    for t in range(L, 2 * L):
        tl.store(in_ptr + dest_bc + t, 0.0)


@triton.jit
def dft_one_k_kernel(in_ptr, out_real_ptr, out_imag_ptr,
                     B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, two_L: tl.constexpr):
    # One Triton program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)
    # Output flattened buffer is (B*C, L+1)
    base_out = b * C * (L + 1) + c * (L + 1)

    # Loop over k = 0..L: compute DFT for this frequency
    for k in range(0, L + 1):
        acc_real = 0.0
        acc_imag = 0.0
        # Sum over t = 0..two_L-1
        for t in range(0, two_L):
            val = tl.load(in_ptr + (b * C * (2 * L) + c * (2 * L)) + t)  # per-(b, c) source buffer
            # angle = 2*pi*k*t / two_L
            angle = (2.0 * 3.141592653589793 * k * t) / two_L
            # real contribution: val * cos(angle); imag contribution: -val * sin(angle)
            acc_real += val * tl.cos(angle)
            acc_imag += -val * tl.sin(angle)

        # Normalize by two_L
        acc_real = acc_real / two_L
        acc_imag = acc_imag / two_L

        # Store normalized real and imag parts
        out_index = base_out + k
        tl.store(out_real_ptr + out_index, acc_real)
        tl.store(out_imag_ptr + out_index, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute real FFT for each (b, c) slice of x of shape (B, C, L), implicit zero-padding to 2*L,
        return real and imaginary parts of length L+1, normalized by 2*L, both float32 tensors of shape (B, C, L+1).
        All computation is performed via Triton kernels launched in forward.
        """
        assert x.dim() == 3, "Input must be of shape (B, C, L)"
        B, C, L = x.shape
        two_L = 2 * L

        # Ensure float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # Allocate flattened padded input per (b, c): buffer of shape (B*C, 2*L)
        in_buf = torch.empty((B * C, two_L), dtype=torch.float32, device=x.device)

        # Launch pad_input_kernel: one program per (b, c)
        grid_pad = (B, C)
        pad_input_kernel[grid_pad](
            x_f32.view(B * C, L),  # view x as (B*C, L)
            in_buf,
            B=B, C=C, L=L
        )

        # Allocate output flattened buffers: shape (B*C, L+1)
        out_real = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)

        # Launch dft_one_k_kernel: one program per (b, c), compute for k in [0..L]
        grid_dft = (B, C)
        dft_one_k_kernel[grid_dft](
            in_buf, out_real, out_imag,
            B=B, C=C, L=L, two_L=two_L
        )

        # Reshape to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
