import torch
import triton
import triton.language as tl


@triton.jit
def real_dft_per_bc_k_kernel(
    x_ptr,         # *float32, pointer to flattened padded input across all (b,c) slices
    out_real_ptr,  # *float32, pointer to output real part flattened as ((B*C)*(L+1))
    out_imag_ptr,  # *float32, pointer to output imag part flattened as ((B*C)*(L+1))
    two_L: tl.constexpr,     # int, length of padded input (2*L)
    L_out: tl.constexpr,     # int, output length (L+1)
    k: tl.constexpr,         # int, current frequency index
):
    # Grid:
    # axis 0: bc_id in [0, B*C)
    # axis 1: k in [0, L]
    pid_bc = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Safety check
    if pid_bc >= (B * C) or pid_k >= L_out:
        return

    # Each (b,c) slice occupies 'two_L' contiguous elements in x_ptr
    base = pid_bc * two_L

    # Accumulator (float32) for this k
    acc_real = 0.0
    acc_imag = 0.0

    # Loop over t in [0..2*L-1]
    two_pi = 2.0 * 3.141592653589793

    for t in range(0, two_L):
        val = tl.load(x_ptr + base + t)
        angle = two_pi * (float(k) * float(t)) / float(two_L)
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)
        # For real input, rfft(x)[k] = sum_{t} x[t] * (cos - i sin)
        acc_real += val * cos_term
        acc_imag += -val * sin_term  # negative sign due to -i

    # Normalize by 2*L
    norm = float(two_L)
    acc_real = acc_real / norm
    acc_imag = 0.0  # Imaginary part is zero for real inputs, but we store both as required

    # Compute flat output index for (bc_id, k)
    out_index = pid_bc * L_out + pid_k

    # Store results
    tl.store(out_real_ptr + out_index, acc_real)
    tl.store(out_imag_ptr + out_index, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure input is float32 and contiguous
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        two_L = 2 * L
        L_out = L + 1

        # Prepare input: for each (b,c), pad with zeros to length 2*L
        # Create a single 1D buffer that contains all (b,c) slices back-to-back.
        # Each (b,c) slice has length L, followed by L zeros.
        x_padded = torch.empty((B, C, two_L), dtype=torch.float32, device=x.device)
        x_padded[:, :, :L] = x
        x_padded[:, :, L:] = 0.0
        # Flatten to 1D for Triton: length = (B*C) * (2*L)
        x_flat = x_padded.reshape(-1)

        # Allocate outputs
        out_real = torch.empty((B * C * L_out), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C * L_out), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: grid over (B*C, L+1)
        grid = (B * C, L_out)
        real_dft_per_bc_k_kernel[grid](
            x_flat,
            out_real,
            out_imag,
            two_L=two_L,
            L_out=L_out,
        )

        # Reshape outputs to (B, C, L+1)
        x_freq_real = out_real.view(B, C, L_out).contiguous()
        x_freq_imag = out_imag.view(B, C, L_out).contiguous()

        # Normalize by 2*L (already done in kernel)
        # Return real and imaginary parts as float32
        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
